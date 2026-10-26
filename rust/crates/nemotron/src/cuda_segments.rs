//! Nemotron's prompt as staggered segments (cuda/segments.zig): each
//! segment is a whole serial chunk, so the bits match.
//!
//! Port of `zig/src/families/nemotron/cuda_segments.zig` (110 lines) —
//! the Hooks reach the engine through a raw pointer with the same
//! invariants the engine's `with_head` documents (the drive's Fam is
//! called from inside the runner, which only holds the fam).

use tensorfold_cuda::segments::{self, Runner, Wait};
use tensorfold_cuda::Stream;

use crate::cuda_engine::{Cancel, Engine};
use crate::cuda_forward::{Forward, Walk};
use crate::cuda_kernels::Ops;
use crate::cuda_state::{Buffers, PREFILL_ROWS};
use crate::error::Error;

pub use segments::MAX;

/// The side streams, events and extra scratch sets a call of up to `n`
/// segments uses (made on the first such prompt). Upstream: `Segments`
/// (cuda_segments.zig:17).
pub struct Segments<'a, 'd> {
    pub runner: Runner<'d>,
    /// The per-segment scratch, each sharing the engine's caches and
    /// state (follow).
    pub sibs: [Option<Buffers<'a, 'd>>; MAX - 1],
    pub n: usize,
}

impl<'a, 'd> Segments<'a, 'd> {
    /// The runner's address, for the engine's split borrows.
    pub(crate) fn runner_ptr(&self) -> *const Runner<'d> {
        &self.runner
    }

    /// Upstream: `init` (cuda_segments.zig:22).
    pub fn init(e: &Engine<'a, 'd>, n: usize) -> Result<Segments<'a, 'd>, Error> {
        let runner = Runner::init(e.ctx.d, n)?;
        let mut s = Segments {
            runner,
            sibs: [const { None }; MAX - 1],
            n: 1,
        };
        while s.n < n {
            s.sibs[s.n - 1] = Some(Buffers::sibling(&e.b, e.ctx.d, e.c, e.nch)?);
            s.n += 1;
        }
        Ok(s)
    }
}

/// One segment: a forward on its stream and scratch, and where its
/// chunk stands. Upstream: `Seg` (cuda_segments.zig:37) — the port
/// keeps the buffer view and the walk; the Forward is rebuilt at each
/// hook from the engine, the view and the segment's stream.
struct SegPart<'a, 'd> {
    bufs: *const Buffers<'a, 'd>,
    walk: Walk,
    stream: Option<&'a Stream<'d>>,
}

/// Nemotron's hooks, Forward.chunk's parts; every block waits ahead of
/// its mixer (in-place states and cache keys). Upstream: `Hooks`
/// (cuda_segments.zig:40) — the engine pointer is the drive's bridge
/// (SAFETY: the Hooks live only inside `prefill`, the engine is alive
/// throughout, and no Hooks field aliases an Engine field; the raw
/// buffer views point into the engine's `b` and its segments' siblings,
/// alive as long as `e.seg` is).
struct Hooks<'a, 'd> {
    e: *mut Engine<'a, 'd>,
    list: [Option<SegPart<'a, 'd>>; MAX],
    /// The drive's segment count, upstream's `list.len`.
    live: usize,
}

/// The staggered prompt from a reset state, `parts` chunks a call
/// (`cancel` asked between calls); leaves the head and prompt buffers
/// as serial does. Upstream: `prefill` (cuda_segments.zig:65).
pub fn prefill<'a, 'd>(
    e: &mut Engine<'a, 'd>,
    prompt: &[u32],
    head: bool,
    parts: usize,
    mut cancel: Option<&mut Cancel<'_>>,
) -> Result<(), Error> {
    // SAFETY: the segments' n read once, their scratch alive in e.seg.
    let seg_n = e.seg.as_ref().map_or(1, |s| s.n);
    if parts < 2 || parts > seg_n {
        return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // Segments
    }
    let r = PREFILL_ROWS;
    let mut at: usize = 0;
    let mut last: usize = 0;
    let mut last_rows: usize = 0;
    // the buffer views: [0] the engine's own, [k] the k-1 sibling's
    let mut bufs: [*const Buffers<'a, 'd>; MAX] = [std::ptr::null(); MAX];
    while at < prompt.len() {
        if Cancel::now(cancel.as_deref_mut()) {
            return Err(Error::Cancelled);
        }
        e.before_chunk(cancel.as_deref_mut())?;
        let c = segments::chunked(prompt.len() - at, r, parts);
        {
            // the siblings take the engine's sequence buffers for this
            // chunk (they move when the engine binds another sequence)
            let (seg, b) = e.split_seg_buffers();
            for k in 1..parts {
                seg.sibs[k - 1].as_mut().unwrap().follow(b);
            }
        }
        bufs[0] = &e.b as *const Buffers<'a, 'd>;
        #[allow(clippy::needless_range_loop)] // upstream's own range loop
        for k in 1..parts {
            // SAFETY: the sibling stands (filled at Segments::init).
            bufs[k] = e.seg.as_ref().unwrap().sibs[k - 1].as_ref().unwrap() as *const Buffers<'a, 'd>;
        }
        e.copied.synchronize()?;
        // the chunk's ids staged for the uploads (the caller's stream
        // reads them before the fork, as the pinned span does serially)
        let host: Vec<u32> = prompt[at..at + c.rows].to_vec();
        // every segment's ids go up on the caller's stream, ahead of the
        // call's fork
        let mut part_streams: [Option<*const Stream<'d>>; MAX] = [None; MAX];
        {
            let streams = e.segment_streams(c.parts);
            for k in 0..c.parts {
                let rows = segments::chunk_rows(c.rows, r, k);
                let ids: &[u8] = u32_bytes(&host[k * r..k * r + rows]);
                let ops = Ops {
                    k: &e.k,
                    s: &e.stream,
                };
                ops.upload(unsafe { (*bufs[k]).p_ids }, ids)?;
                part_streams[k] = Some(streams[k]);
            }
        }
        e.copied.record(&e.stream)?;
        let pos0 = e.pos;
        {
            let e_ptr: *mut Engine<'a, 'd> = e;
            let runner = unsafe { &*e_ptr }.seg_runner();
            let mut hooks = Hooks {
                e: e_ptr,
                list: [const { None }; MAX],
                live: c.parts,
            };
            let _live = hooks.live; // the drive's length, upstream's list.len
            for k in 0..c.parts {
                let rows = segments::chunk_rows(c.rows, r, k);
                // SAFETY: the segment's stream pointer from the runner.
                let stream = unsafe { &*part_streams[k].unwrap() };
                hooks.list[k] = Some(SegPart {
                    bufs: bufs[k],
                    walk: Walk::new(rows, pos0 + k * r),
                    stream: Some(stream),
                });
            }
            let blocks = unsafe { &*e_ptr }.w.blocks.len();
            // SAFETY: the drive — the hooks' bridge documented above;
            // the runner's own state is the drive's, only read.
            unsafe { &*(runner as *const Runner<'d>) }
                .run::<_, Error>(&unsafe { &*e_ptr }.stream, c.parts, blocks, &mut hooks)?;
        }
        for k in 0..c.parts {
            let start = at + k * r;
            let known = segments::chunk_rows(c.rows, r, k).min(prompt.len() - 1 - start);
            if head && known > 0 {
                let hidden = e.b.p_hidden;
                let tokens: Vec<u32> = prompt[start + 1..start + 1 + known].to_vec();
                e.with_head(move |h, e| h.absorb(e, hidden, &tokens))
                    .ok_or(Error::NoMtpHead)??;
            }
        }
        last = c.parts - 1;
        last_rows = segments::chunk_rows(c.rows, r, last);
        e.pos += c.rows;
        at += c.rows;
    }
    if last == 0 {
        return Ok(());
    }
    let o = e.ops();
    o.copy(e.b.p_hidden, unsafe { (*bufs[last]).p_hidden }, last_rows * e.c.hidden * 2)?;
    o.copy(e.b.p_logits, unsafe { (*bufs[last]).p_logits }, e.c.vocab * 2)?;
    o.copy(e.b.p_sampled, unsafe { (*bufs[last]).p_sampled }, 4)?;
    Ok(())
}

/// The u32 slice's own bytes.
fn u32_bytes(w: &[u32]) -> &[u8] {
    // SAFETY: the slice's own bytes, read-only.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), w.len() * 4) }
}

/// The Hooks drive the chunk hooks on their segment's stream.
impl<'a, 'd> tensorfold_cuda::segments::Fam<Error> for Hooks<'a, 'd> {
    fn begin(&mut self, l: &tensorfold_cuda::segments::Lane) -> Result<(), Error> {
        let k = l.k;
        let (bufs, walk, s) = self.part_parts(k);
        let e = self.engine();
        // SAFETY: the runner's segment stream, alive through the drive.
        let s: &Stream<'d> = unsafe { &*s };
        let ops = Ops {
            k: &e.k,
            s,
        };
        // SAFETY: the engine's config read by copy through the drive.
        let c = unsafe { std::ptr::read(&e.c) };
        let b: &Buffers<'a, 'd> = unsafe { &*bufs };
        let f = Forward::init(c, &e.w, b, ops, e.max_len, e.nch, e.sampling.is_some());
        let mut w = walk;
        f.chunk_begin(&mut w)?;
        self.set_walk(k, w);
        Ok(())
    }

    fn wait(&mut self, _i: usize) -> Wait {
        Wait::Mixer
    }

    fn handoff(&mut self, _l: &tensorfold_cuda::segments::Lane, _i: usize) -> Result<(), Error> {
        Ok(())
    }

    fn pre(&mut self, l: &tensorfold_cuda::segments::Lane, i: usize) -> Result<(), Error> {
        let k = l.k;
        let (bufs, walk, s) = self.part_parts(k);
        let e = self.engine();
        // SAFETY: the runner's segment stream, alive through the drive.
        let s: &Stream<'d> = unsafe { &*s };
        let ops = Ops {
            k: &e.k,
            s,
        };
        let c = unsafe { std::ptr::read(&e.c) };
        let b: &Buffers<'a, 'd> = unsafe { &*bufs };
        let f = Forward::init(c, &e.w, b, ops, e.max_len, e.nch, e.sampling.is_some());
        let mut w = walk;
        f.chunk_pre(&mut w, i)?;
        self.set_walk(k, w);
        Ok(())
    }

    fn mixer(&mut self, l: &tensorfold_cuda::segments::Lane, i: usize) -> Result<(), Error> {
        let k = l.k;
        let (bufs, walk, s) = self.part_parts(k);
        let e = self.engine();
        // SAFETY: the runner's segment stream, alive through the drive.
        let s: &Stream<'d> = unsafe { &*s };
        let ops = Ops {
            k: &e.k,
            s,
        };
        let c = unsafe { std::ptr::read(&e.c) };
        let b: &Buffers<'a, 'd> = unsafe { &*bufs };
        let f = Forward::init(c, &e.w, b, ops, e.max_len, e.nch, e.sampling.is_some());
        let mut w = walk;
        f.chunk_mixer(&mut w, i)?;
        self.set_walk(k, w);
        Ok(())
    }

    fn post(&mut self, l: &tensorfold_cuda::segments::Lane, i: usize) -> Result<(), Error> {
        let k = l.k;
        let (bufs, walk, s) = self.part_parts(k);
        let e = self.engine();
        // SAFETY: the runner's segment stream, alive through the drive.
        let s: &Stream<'d> = unsafe { &*s };
        let ops = Ops {
            k: &e.k,
            s,
        };
        let c = unsafe { std::ptr::read(&e.c) };
        let b: &Buffers<'a, 'd> = unsafe { &*bufs };
        let f = Forward::init(c, &e.w, b, ops, e.max_len, e.nch, e.sampling.is_some());
        let mut w = walk;
        f.chunk_post(&mut w, i)?;
        self.set_walk(k, w);
        Ok(())
    }

    fn finish(&mut self, l: &tensorfold_cuda::segments::Lane) -> Result<(), Error> {
        let k = l.k;
        let (bufs, walk, s) = self.part_parts(k);
        let e = self.engine();
        // SAFETY: the runner's segment stream, alive through the drive.
        let s: &Stream<'d> = unsafe { &*s };
        let ops = Ops {
            k: &e.k,
            s,
        };
        let c = unsafe { std::ptr::read(&e.c) };
        let b: &Buffers<'a, 'd> = unsafe { &*bufs };
        let f = Forward::init(c, &e.w, b, ops, e.max_len, e.nch, e.sampling.is_some());
        let mut w = walk;
        f.chunk_finish(&mut w)?;
        self.set_walk(k, w);
        Ok(())
    }
}

impl<'a, 'd> Hooks<'a, 'd> {
    fn engine(&mut self) -> &mut Engine<'a, 'd> {
        // SAFETY: prefill's scope owns the engine alive.
        unsafe { &mut *self.e }
    }

    /// The segment's buffer view, its walk and its stream — both raw
    /// pointers (the drive's scope keeps them alive; the borrows end
    /// where they are taken).
    fn part_parts(&self, k: usize) -> (*const Buffers<'a, 'd>, Walk, *const Stream<'d>) {
        let p = self.list[k].as_ref().unwrap();
        (p.bufs, p.walk.clone(), p.stream.unwrap())
    }

    #[allow(clippy::needless_range_loop)] // upstream's own range loop
    fn set_walk(&mut self, k: usize, w: Walk) {
        if let Some(p) = self.list[k].as_mut() {
            p.walk = w;
        }
    }


}
