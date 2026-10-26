//! Nemotron CUDA streams own their sequences and share window rounds.
//!
//! Port of `zig/src/families/nemotron/cuda_lanes.zig` (332 lines) — the
//! Backend trait impl over the engine, the stream map keyed by the
//! lane core's raw stream pointers (the crate's own invariant).

use std::collections::HashMap;

use tensorfold_cuda::HostBuffer;
use tensorfold_lanes::backend::{Backend, DraftRequest, Feed, Verified, Window};
use tensorfold_lanes::stream::Stream;
use tensorfold_lanes::{Cost, LogprobRow, Model};
use tensorfold_lanes::row_words;

use crate::cuda_engine as engine;
use crate::cuda_engine::{Engine, Keep, MAX_STREAMS, Shared};
use crate::cuda_kernels::Ops;
use crate::cuda_mtp::MAX_CHAIN;
use crate::cuda_reuse;
use crate::cuda_state::{Seq, MAX_ROWS, PREFILL_ROWS};
use crate::error::Error as Family;

/// The vtable's errors: the family's refusals carried as upstream's
/// anyerror rides the vtable — the port's lanes::Error::Backend holds
/// their words.
type Error = tensorfold_lanes::Error;

/// The driver's errors mapped (the orphan rule keeps this a fn).
fn carried_cuda(e: tensorfold_cuda::Error) -> Error {
    Error::Backend(e.to_string())
}

impl From<Family> for Error {
    fn from(e: Family) -> Error {
        Error::Backend(e.to_string())
    }
}

fn carried(e: Family) -> Error {
    Error::Backend(e.to_string())
}

/// A family call, mapped into the vtable's error.
macro_rules! carried {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Err(carried(e)),
        }
    };
}
use crate::config;

/// Host-mapped words for every row a shared window holds, then the
/// first token's row.
const ROW_SLOTS: usize = MAX_STREAMS * MAX_ROWS + 1;

/// First tokens a handle names (the prompt's draw is on the host once
/// prefill returns).
const RING: usize = 1024;

/// A stream's sequence (`own`: the engine's, the graphs' buffers) and
/// its first token's logprob row from prefill. Upstream: `Lane`
/// (cuda_lanes.zig:26).
struct Lane<'a, 'd> {
    seq: *mut Seq<'a, 'd>,
    own: bool,
    pending_rows: Option<usize>,
    first: Option<LogprobRow>,
}

impl<'a, 'd> Lane<'a, 'd> {
    fn new(seq: *mut Seq<'a, 'd>, own: bool) -> Lane<'a, 'd> {
        Lane {
            seq,
            own,
            pending_rows: None,
            first: None,
        }
    }
}

/// Upstream: `Cuda` (cuda_lanes.zig:28).
pub struct Cuda<'a, 'd> {
    /// The engine, as `with_head`'s bridge documents it (boxed by its
    /// owner — the Loaded —, alive as long as this backend; no Cuda
    /// field aliases an Engine field).
    e: *mut Engine<'a, 'd>,
    pub has_head: bool,
    /// Keyed by the lane core's stream pointers (its own invariant:
    /// live, boxed streams).
    /// The lane core's stream pointers as raw keys (its own invariant:
    /// live, boxed streams).
    lanes: HashMap<usize, Lane<'a, 'd>>,
    /// No stream holds the engine's own sequence.
    own_free: bool,
    drawn: [u32; RING],
    next: u64,
    /// Each window's held drafts read back (state.max_rows words a
    /// stream).
    pinned: HostBuffer<'d>,
    /// Logprob rows the kernel writes straight to the host.
    rows_out: HostBuffer<'d>,
    rows_dev: u64,
    costs: [Cost; MAX_ROWS],
    cost_count: usize,
    pub mtp_ms: f64,
    /// A lone stream's depth rule prices its rounds by these.
    pub measured: Option<tensorfold_core::draft_depth::Costs>,
}

impl<'a, 'd> Cuda<'a, 'd> {
    /// The engine behind the bridge (SAFETY: the Loaded's Box, alive
    /// as long as this backend).
    pub(crate) fn e(&self) -> &Engine<'a, 'd> {
        unsafe { &*self.e }
    }

    /// The engine, mutable, behind the bridge.
    pub(crate) fn e_mut(&mut self) -> &mut Engine<'a, 'd> {
        unsafe { &mut *self.e }
    }

    /// Upstream: `init` (cuda_lanes.zig:44). The raw engine pointer is
    /// the Loaded's bridge (the same invariant `with_head` documents);
    /// the port keeps it in the signature rather than an unsafe fn.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // the raw shape is the documented invariant
    pub fn init(e: *mut Engine<'a, 'd>) -> Result<Cuda<'a, 'd>, Family> {
        // SAFETY: the Loaded's boxed engine, alive as long as this
        // backend (the bridge's invariant).
        let engine = unsafe { &*e };
        let pinned = HostBuffer::alloc(engine.ctx.d, MAX_STREAMS * MAX_ROWS * 4)?;
        let rows_out = HostBuffer::alloc_mapped(engine.ctx.d, ROW_SLOTS * row_words * 4)?;
        let rows_dev = rows_out.device()?;
        let has_head = engine.head.is_some();
        Ok(Cuda {
            e,
            has_head,
            lanes: HashMap::new(),
            own_free: true,
            drawn: [0; RING],
            next: 0,
            pinned,
            rows_out,
            rows_dev,
            costs: [Cost {
                width: 0,
                ms: 0.0,
            }; MAX_ROWS],
            cost_count: 0,
            mtp_ms: 0.0,
            measured: None,
        })
    }

    /// Upstream: `deinit` (cuda_lanes.zig:52) — the port's Drop frees
    /// the sequences it owns.
    pub fn deinit(&mut self) {
        let lanes: Vec<Lane<'a, 'd>> = self.lanes.drain().map(|(_, l)| l).collect();
        for l in lanes {
            self.drop(l);
        }
    }

    /// Upstream: `backend` (cuda_lanes.zig:60) — the port's Backend is
    /// a trait; the impl block below is the vtable.
    pub fn facts(&self) -> Model<'_> {
        let drafting = self.has_head;
        Model {
            exact_width: if drafting { MAX_ROWS as u32 } else { 1 },
            gpu_tokens: false,
            mtp: drafting,
            speculate: drafting,
            speculate_early: false,
            draft_prior: &config::DRAFT_PRIOR,
            drafts: MAX_CHAIN as u32,
            window_costs: &self.costs[..self.cost_count],
            mtp_step_ms: self.mtp_ms,
            hidden_rows: drafting, // several streams' windows share a forward
            batch_rows: MAX_ROWS as u32,
            max_streams: if drafting { MAX_STREAMS as u32 } else { 1 },
            ..Model::default()
        }
    }

    /// costs.measure on the engine's own sequence: windows of 1 to 16
    /// rows and a head level, in ms. Upstream: `measure`
    /// (cuda_lanes.zig:94).
    pub fn measure(&mut self, model_dir: &std::path::Path) -> Result<(), Family> {
        if !self.has_head {
            return Ok(());
        }
        // SAFETY: the own view's address from behind the bridge.
        // SAFETY: the own view's address from behind the bridge.
        let own = &mut unsafe { &mut *self.e }.own as *mut Seq<'a, 'd>;
        self.e_mut().bind(own);
        let c = crate::cuda_costs::measure(self.e_mut(), model_dir)?;
        self.cost_count = 0;
        for w in 1..c.rows + 1 {
            self.costs[self.cost_count] = Cost {
                width: w as u32,
                ms: c.verify[w],
            };
            self.cost_count += 1;
        }
        self.mtp_ms = c.level;
        self.measured = Some(c);
        Ok(())
    }

    /// A lone driver's hand-over: the head absorbs the last `kept`
    /// rows; the stream stands as a lane round leaves it. Upstream:
    /// `handOver` (cuda_lanes.zig:108).
    pub fn hand_over(&mut self, s: &mut Stream<'_>, kept: usize) -> Result<(), Family> {
        let l = match self.lanes.get_mut(&(s as *mut Stream<'_> as usize)) {
            Some(l) => l,
            None => return Err(Family::UnknownStream),
        };
        l.pending_rows = None;
        if self.has_head {
            self.e_mut()
                .with_head(|h, e| h.begin(e, kept))
                .ok_or(Family::NoMtpHead)??;
            self.e_mut()
                .with_head(|h, e| h.launch(e, 0))
                .ok_or(Family::NoMtpHead)??;
        }
        s.cache_len = self.e().pos as u64;
        let pending = s.context[s.context.len() - 1];
        s.pending = Some(pending);
        Ok(())
    }

    /// Bind the lane and commit its kept rows, accounting for
    /// shared-round drafting. Upstream: `bindLane` (cuda_lanes.zig:120).
    fn bind_lane(
        &mut self,
        s: &mut Stream<'_>,
        kept: Option<usize>,
    ) -> Result<Lane<'a, 'd>, Family> {
        let (seq, pending) = match self.lanes.get(&(s as *mut Stream<'_> as usize)) {
            Some(l) => (l.seq, l.pending_rows),
            None => return Err(Family::UnknownStream),
        };
        self.e_mut().bind(seq);
        if let Some(rows) = pending {
            self.e_mut().commit(kept.unwrap_or(rows))?;
        }
        self.lanes
            .get_mut(&(s as *mut Stream<'_> as usize))
            .unwrap()
            .pending_rows = None;
        Ok(Lane {
            seq,
            own: false,
            pending_rows: None,
            first: None,
        })
    }

    fn take(&mut self, token: u32) -> u64 {
        let h = self.next;
        self.drawn[(h % RING as u64) as usize] = token;
        self.next += 1;
        h
    }

    fn value(&self, feed: Feed) -> u32 {
        match feed {
            Feed::Handle(h) => self.drawn[(h % RING as u64) as usize],
            Feed::Value(v) => v,
        }
    }

    /// The stream's kept state on its bound sequence: where its prompt
    /// pass starts (0: none, off the grid, or failed). Upstream:
    /// `restoreKept` (cuda_lanes.zig:157).
    fn restore_kept(&mut self, s: &mut Stream<'_>, len: usize) -> usize {
        let saved = s.reuse.saved;
        if saved.is_null() {
            return 0;
        }
        if !crate::cuda_prompt_grid::resumable(s.reuse.at as usize, len, PREFILL_ROWS) {
            s.reuse_failed = true;
            return 0;
        }
        let mut target = cuda_reuse::Target {
            e: self.e,
        };
        let at = s.reuse.at as usize;
        // SAFETY: the saved copy the host made (its own boxed copy),
        // restored here as upstream's anyopaque pair does.
        if unsafe { cuda_reuse::restore(&mut target, saved, at as u32) }.is_err() {
            s.reuse_failed = true;
            return 0;
        }
        s.cached = s.reuse.at;
        at
    }

    /// A window's rows from the words its kernel wrote at slot `row0`
    /// (after the stream synchronized). Upstream: `readRows`
    /// (cuda_lanes.zig:258).
    fn read_rows(&self, o: &mut Verified<'_>, row0: usize, k: u8) {
        // SAFETY: the mapped host rows, aligned; the view is brief.
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(self.rows_out.slice().as_ptr().cast(), ROW_SLOTS * row_words)
        };
        for (i, r) in o.rows.iter_mut().enumerate() {
            let slot = row0 + i;
            *r = LogprobRow::from_words(
                &words[slot * row_words..(slot + 1) * row_words],
                o.sampled[i],
                k,
            );
        }
    }

    /// Several windows in one forward (Engine.verifyShared), each
    /// stream's rows on its own sequence. Upstream: `verifyShared`
    /// (cuda_lanes.zig:264).
    fn verify_shared(&mut self, windows: &[Window<'_>], out: &mut [Verified<'_>]) -> Result<(), Family> {
        let e: *mut Engine<'a, 'd> = self.e;
        if windows.len() > MAX_STREAMS {
            return Err(Family::WindowTooWide);
        }
        let mut ids: Vec<Vec<u32>> = (0..MAX_STREAMS).map(|_| vec![0u32; MAX_ROWS]).collect();
        // the ids' raw spans: the engine takes them as slices whose
        // borrow the engine's own scope holds (the parts out of the
        // map's borrows)
        let mut parts: Vec<Shared<'a, 'd>> = Vec::with_capacity(windows.len());
        let mut id_spans: [(*const u32, usize); MAX_STREAMS] = [(std::ptr::null(), 0); MAX_STREAMS];
        for (k, w) in windows.iter().enumerate() {
            if w.parents.is_some() {
                return Err(Family::Cuda(tensorfold_cuda::Error::Invalid)); // TreesNotBuilt
            }
            // SAFETY: the window's stream pointer, live under the lane
            // core's invariant.
            let stream: &mut Stream<'_> = unsafe { &mut *w.stream };
            self.bind_lane(stream, None)?;
            let l_seq = self.lanes[&(w.stream as usize)].seq;
            for (r, p) in w.positions.iter().enumerate() {
                if *p != unsafe { &*e }.pos as u64 + 1 + r as u64 {
                    return Err(Family::Cuda(tensorfold_cuda::Error::Invalid)); // PositionMismatch
                }
            }
            ids[k][0] = w.pending;
            ids[k][1..1 + w.tokens.len()].copy_from_slice(w.tokens);
            let rows = w.rows();
            let id_len = 1 + w.tokens.len();
            id_spans[k] = (ids[k].as_ptr(), id_len);
            parts.push(Shared {
                seq: l_seq,
                ids: unsafe {
                    // SAFETY: the span points into `ids`, alive
                    // through the verify_shared call below.
                    std::slice::from_raw_parts(id_spans[k].0, id_spans[k].1)
                },
                rows,
            });
        }
        // SAFETY: the engine behind the raw pointer, as the drive's
        // bridges document (the parts' ids live in this scope).
        unsafe { &mut *e }.verify_shared(&parts)?;
        let held: &[u32] = pinned_words(&self.pinned);
        let mut row0: usize = 0;
        for (k, (w, o)) in windows.iter().zip(out.iter_mut()).enumerate() {
            let seq = parts[k].seq;
            unsafe { &mut *e }.bind(seq);
            if w.held > 0 {
                // SAFETY: the pinned words' bytes, held for the copy.
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        (self.pinned.slice().as_ptr() as *mut u32).add(k * MAX_ROWS).cast(),
                        (rows_of(w) - 1) * 4,
                    )
                };
                unsafe { &mut *e }.ops().download(bytes, unsafe { &*e }.b.ids + 4)?;
            }
            // the shared scratch holds every part's logits, part k's
            // from its first row; its picks are its own
            if !o.rows.is_empty() {
                unsafe { &mut *e }.ops().logprob_rows(
                    unsafe { &*e }.b.logits + row0 as u64 * unsafe { &*e }.c.vocab as u64 * 2,
                    unsafe { &*e }.c.vocab,
                    unsafe { &*e }.b.sampled,
                    k as u8,
                    self.rows_dev + row0 as u64 * row_words as u64 * 4,
                    rows_of(w),
                )?;
            }
            row0 += rows_of(w);
        }
        unsafe { &mut *e }.stream.synchronize()?;
        let mut row0: usize = 0;
        for (k, (w, o)) in windows.iter().zip(out.iter_mut()).enumerate() {
            let tokens = unsafe { &mut *e }.shared_tokens(k, rows_of(w))?;
            o.sampled.copy_from_slice(&tokens);
            let drafts: &[u32] = if w.held > 0 {
                &held[k * MAX_ROWS..k * MAX_ROWS + rows_of(w) - 1]
            } else {
                w.tokens
            };
            o.drafts.copy_from_slice(drafts);
            if !o.rows.is_empty() {
                self.read_rows(o, row0, k as u8);
            }
            row0 += rows_of(w);
            self.lanes.get_mut(&(w.stream as usize)).unwrap().pending_rows = Some(rows_of(w));
        }
        Ok(())
    }

    fn drop(&mut self, l: Lane<'a, 'd>) {
        if l.own {
            self.own_free = true;
        } else {
            self.e_mut().free_seq(l.seq);
        }
    }
}

impl Drop for Cuda<'_, '_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

fn rows_of(w: &Window<'_>) -> usize {
    w.rows()
}

/// The pinned words' bytes at `at`, `n` words, mutable.
fn pinned_words_mut<'p>(pinned: &'p mut HostBuffer<'_>, at: usize, n: usize) -> &'p mut [u8] {
    // SAFETY: the words' own bytes, the span's.
    unsafe {
        std::slice::from_raw_parts_mut(
            (pinned.slice_mut().as_mut_ptr() as *mut u32).add(at).cast(),
            n * 4,
        )
    }
}

/// The pinned words, read-only.
fn pinned_words<'p>(pinned: &'p HostBuffer<'_>) -> &'p [u32] {
    // SAFETY: the aligned pinned words; the view is brief.
    unsafe {
        std::slice::from_raw_parts(pinned.slice().as_ptr().cast(), pinned.slice().len() / 4)
    }
}

/// The Backend vtable (upstream: the fns behind `backend()`).
// The Backend trait's raw stream pointers are the port's documented
// invariant (the trait doc: live, boxed streams under the crate's
// invariants); the impl dereferences them exactly as the trait's
// default `opening` does.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl<'a, 'd> Backend for Cuda<'a, 'd> {
    fn prefill(&mut self, s: *mut Stream<'_>) -> Result<(), Error> {
        // SAFETY: the lane core's stream pointer, live under its
        // invariant.
        let stream: &mut Stream<'_> = unsafe { &mut *s };
        let ids: Vec<u32> = stream.prompt().to_vec();
        if ids.is_empty() || ids.len() + stream.max_new as usize + MAX_ROWS > self.e().max_len {
            return Err(Error::Backend("the prompt does not leave room for the reply".into()));
        }
        if let Some(l) = self.lanes.get(&(s as usize)) {
            // the existing lane's sequence drops before its new one
            // stands
            let l = Lane {
                seq: l.seq,
                own: l.own,
                pending_rows: None,
                first: None,
            };
            self.lanes.remove(&(s as usize));
            self.drop(l);
        }
        // the engine's own sequence while it is free: its rounds replay
        // the captured graphs
        let own = self.own_free && self.e().serial.is_some();
        // SAFETY: the own view's address from behind the bridge.
        let seq: *mut Seq<'a, 'd> = if own {
            &mut unsafe { &mut *self.e }.own
        } else {
            #[allow(clippy::question_mark)] // the error maps through the vtable's shape
            match self.e_mut().new_seq() {
                Ok(p) => p,
                Err(e) => return Err(e.into()),
            }
        };
        self.lanes.insert(s as usize, Lane::new(seq, own));
        if own {
            self.own_free = false;
        }
        self.e_mut().bind(seq);
        let sampling = stream.sampling;
        self.e_mut().set_sampling(sampling)?;
        stream.cached = 0;
        let from = self.restore_kept(stream, ids.len());
        let has_head = self.has_head;
        // the keep hook copies the stream's kept states as the pass
        // stands at each mark
        let marks = stream.reuse.marks;
        // the keep hook over the raw stream pointer (upstream's ctx)
        let mut keep_call = move |at: u32| {
            // SAFETY: the lane core's stream pointer, live under its
            // invariant through the pass.
            let stream: &mut Stream<'_> = unsafe { &mut *s };
            if let Some(hook) = stream.reuse.hook.as_mut() {
                hook(s, at);
            }
        };
        let mut keep: Option<Keep<'_>> = if marks.is_empty() {
            None
        } else {
            Some(Keep {
                at: marks,
                call: &mut keep_call,
            })
        };
        // the cancel adapter: the lanes' CancelCheck as the engine's
        // Cancel (a boxed FnMut over the check)
        // the box drops before the check moves back into the stream
        #[allow(unused)]
        struct CancelBoxTag;
        let mut cancel_taken = stream.cancel_check.take();
        let mut cancel_boxed: Option<Box<dyn FnMut() -> bool + '_>> = cancel_taken
            .as_mut()
            .map(|c| Box::new(move || c.is_cancelled()) as Box<dyn FnMut() -> bool + '_>);
        let mut cancel = cancel_boxed
            .as_mut()
            .map(|c| engine::Cancel { check: c.as_mut() });
        let first = self.e_mut().prefill_from(
            &ids,
            from,
            has_head,
            cancel.as_mut(),
            keep.as_mut(),
        );
        // end the adapter's borrows before the check moves back
        #[allow(unused_assignments)] // the None-assigns end the borrows
        {
            cancel = None;
            cancel_boxed = None;
        }
        stream.cancel_check = cancel_taken;
        let first = carried!(first);
        // the head's first draft reads the prompt's last row (its
        // hidden row waits where a window's would)
        let last = (ids.len() - 1) % PREFILL_ROWS; // a resumed pass ends on the same grid
        let hidden = self.e().b.p_hidden + last as u64 * self.e().c.hidden as u64 * 2;
        self.e()
            .ops()
            .copy(self.e().b.hidden, hidden, self.e().c.hidden * 2)?;
        self.take(first);
        if let Some(k) = stream.logprobs {
            let at = (ROW_SLOTS - 1) * row_words;
            self.e().ops().logprob_rows(
                self.e().b.p_logits,
                self.e().c.vocab,
                self.e().b.p_sampled,
                k,
                self.rows_dev + at as u64 * 4,
                1,
            )?;
            self.e().stream.synchronize().map_err(carried_cuda)?;
            // SAFETY: the mapped rows, aligned; the view is brief.
            let words: &[u32] = unsafe {
                std::slice::from_raw_parts(self.rows_out.slice().as_ptr().cast(), ROW_SLOTS * row_words)
            };
            let row = LogprobRow::from_words(&words[at..at + row_words], first, k);
            self.lanes.get_mut(&(s as usize)).unwrap().first = Some(row);
        }
        Ok(())
    }

    fn first(&mut self, s: *mut Stream<'_>, position: u64) -> Result<u64, Error> {
        // SAFETY: the stream pointer.
        let stream: &Stream<'_> = unsafe { &*s };
        if position != stream.prompt_len as u64 {
            return Err(Error::PositionMismatch);
        }
        Ok(self.next - 1)
    }

    fn queue(&mut self, _: *mut Stream<'_>, _: Feed, _: u64) -> Result<u64, Error> {
        Err(Error::Backend("the backend does not pipeline".into()))
    }

    fn read(&mut self, handle: u64) -> Result<u32, Error> {
        if handle >= self.next || self.next - handle > RING as u64 {
            return Err(Error::Backend("no drawn token at that handle".into()));
        }
        Ok(self.drawn[(handle % RING as u64) as usize])
    }

    fn verify(&mut self, windows: &[Window<'_>], out: &mut [Verified<'_>]) -> Result<(), Error> {
        if windows.len() > 1 {
            return self.verify_shared(windows, out).map_err(Error::from);
        }
        let w = &windows[0];
        if w.parents.is_some() {
            return Err(Error::Backend("the backend does not build trees".into()));
        }
        // SAFETY: the window's stream.
        let stream: &mut Stream<'_> = unsafe { &mut *w.stream };
        carried!(self.bind_lane(stream, None));
        let rows = w.rows();
        if rows > MAX_ROWS {
            return Err(Error::Backend("the window's rows are too wide".into()));
        }
        for (r, p) in w.positions.iter().enumerate() {
            if *p != self.e().pos as u64 + 1 + r as u64 {
                return Err(Error::PositionMismatch);
            }
        }
        let mut ids = [0u32; MAX_ROWS];
        ids[0] = w.pending;
        ids[1..1 + w.tokens.len()].copy_from_slice(w.tokens);
        self.e_mut().verify(&ids[..1 + w.tokens.len()], rows)?;
        if !out[0].rows.is_empty() {
            let k = stream.logprobs.unwrap();
            self.e().ops().logprob_rows(
                self.e().b.logits,
                self.e().c.vocab,
                self.e().b.sampled,
                k,
                self.rows_dev,
                rows,
            )?;
        }
        let held_len = rows - 1;
        if w.held > 0 {
            // SAFETY: the engine's fields read through the bridge while
            // the pinned span is written — the two share nothing.
            let (ids, k, s) = {
                let e = unsafe { &*self.e };
                (e.b.ids + 4, &e.k, &e.stream)
            };
            let ops = Ops { k, s };
            let bytes: &mut [u8] = pinned_words_mut(&mut self.pinned, 0, held_len);
            ops.download(bytes, ids)?;
        }
        self.e().stream.synchronize().map_err(carried_cuda)?;
        let tokens = self.e_mut().tokens()?;
        out[0].sampled.copy_from_slice(&tokens);
        let held = pinned_words(&self.pinned);
        let drafts: &[u32] = if w.held > 0 {
            &held[..held_len]
        } else {
            w.tokens
        };
        out[0].drafts.copy_from_slice(drafts);
        if !out[0].rows.is_empty() {
            let k = stream.logprobs.unwrap();
            self.read_rows(&mut out[0], 0, k);
        }
        self.lanes.get_mut(&(w.stream as usize)).unwrap().pending_rows = Some(rows);
        Ok(())
    }

    fn keep(&mut self, windows: &[Window<'_>], paths: &[&[u32]]) -> Result<(), Error> {
        for (w, path) in windows.iter().zip(paths.iter()) {
            if path.is_empty() {
                return Err(Error::Backend("a window kept nothing".into()));
            }
            for (i, r) in path.iter().enumerate() {
                if *r as usize != i {
                    return Err(Error::Backend("the backend does not build trees".into()));
                }
            }
            // SAFETY: the window's stream.
            let stream: &mut Stream<'_> = unsafe { &mut *w.stream };
            carried!(self.bind_lane(stream, Some(path.len())));
        }
        Ok(())
    }

    fn draft(&mut self, requests: &[DraftRequest<'_, '_>]) -> Result<(), Error> {
        if !self.has_head {
            return Err(Error::Backend("the checkpoint carries no MTP head".into()));
        }
        for r in requests {
            // SAFETY: the request's stream.
            let stream: &mut Stream<'_> = unsafe { &mut *r.stream };
            carried!(self.bind_lane(stream, r.rows.map(|p| p.len())));
            if r.position != self.e().pos as u64 + 1 {
                return Err(Error::PositionMismatch);
            }
            if let Some(rows) = r.rows {
                for (i, row) in rows.iter().enumerate() {
                    if *row as usize != i {
                        return Err(Error::Backend("the backend does not build trees".into()));
                    }
                }
                self.e_mut()
                    .with_head(|h, e| h.chain(e, r.follow, r.depth as usize))
                    .ok_or(Family::NoMtpHead)??;
            } else {
                let first = match r.first {
                    Some(f) => self.value(f),
                    None => return Err(Error::Backend("the draft request has no first token".into())),
                };
                self.e_mut()
                    .with_head(|h, e| h.chain(e, &[first], r.depth as usize))
                    .ok_or(Family::NoMtpHead)??;
            }
        }
        Ok(())
    }

    fn first_row(&mut self, s: *mut Stream<'_>) -> Result<LogprobRow, Error> {
        match self.lanes.get(&(s as usize)) {
            Some(l) => l.first.ok_or_else(|| Error::Backend("the stream has no first logprob row".into())),
            None => Err(Error::Backend("the backend does not hold this stream".into())),
        }
    }

    fn release(&mut self, s: *mut Stream<'_>) {
        let l = match self.lanes.remove(&(s as usize)) {
            Some(l) => l,
            None => return,
        };
        let _ = self.e().stream.synchronize();
        self.drop(l);
    }
}
