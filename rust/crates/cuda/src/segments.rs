//! Staggered prompt segments on CUDA streams for any family: segment
//! k's mixer waits on k-1's event, the rest overlaps.
//!
//! Port of `zig/src/cuda/segments.zig` (323 lines). The port's `Stream`
//! owns its CUDA object (Zig's is a copied value with manual `deinit`),
//! so `Lane` borrows the stream; the simulation tests borrow the test
//! stub's, the shape of Zig's `undefined` lanes. The stream orders the
//! test tries come from the port's own xorshift — the contract holds
//! for every order, so the generator need not match upstream's
//! DefaultPrng.

use crate::driver::{Driver, Error};
use crate::stream::{Event, Stream};
use tensorfold_lanes::stagger;

pub use tensorfold_lanes::stagger::{chunk_rows, chunked, Call, MAX};
pub use tensorfold_lanes::stagger::Wait;

/// A segment and the stream its work goes on. Upstream: `Lane`
/// (segments.zig:15).
#[derive(Clone, Copy)]
pub struct Lane<'a> {
    pub k: usize,
    pub s: &'a Stream<'a>,
}

/// Side streams and events for up to `n` segments a call, made once and
/// reused by every call. Upstream: `Runner` (segments.zig:18).
pub struct Runner<'d> {
    /// Side streams, one a segment past the first. Upstream: the fixed
    /// `[MAX - 1]Stream` — the port's Stream owns its object, so the
    /// unused tail is None.
    side: [Option<Stream<'d>>; MAX - 1],
    /// handoff[k]: segment k's latest mixer, which segment k+1 waits
    /// for.
    handoff: [Option<Event<'d>>; MAX - 1],
    /// join[k]: side segment k+1's end, which the caller's stream waits
    /// for.
    join: [Option<Event<'d>>; MAX - 1],
    /// The caller's work before a call.
    fork: Option<Event<'d>>,
    n: usize,
}

impl<'d> Runner<'d> {
    /// Upstream: `init` (segments.zig:25) — the fork first, then one
    /// stream and two events a segment past the first; a failure
    /// unwinds what stands.
    pub fn init(d: &'d Driver, n: usize) -> Result<Runner<'d>, Error> {
        if n == 0 || n > MAX {
            return Err(Error::Segments);
        }
        let fork = Event::init(d, false)?;
        let mut r = Runner {
            side: [const { None }; MAX - 1],
            handoff: [const { None }; MAX - 1],
            join: [const { None }; MAX - 1],
            fork: Some(fork),
            n: 1,
        };
        while r.n < n {
            let j = r.n - 1;
            match Stream::init(d, true) {
                Ok(s) => r.side[j] = Some(s),
                Err(e) => {
                    r.deinit();
                    return Err(e);
                }
            }
            match Event::init(d, false) {
                Ok(e) => r.handoff[j] = Some(e),
                Err(e) => {
                    r.deinit(); // drops the side stream that stood
                    return Err(e);
                }
            }
            match Event::init(d, false) {
                Ok(e) => r.join[j] = Some(e),
                Err(e) => {
                    r.deinit(); // drops the side stream and its handoff
                    return Err(e);
                }
            }
            r.n += 1;
        }
        Ok(r)
    }

    /// Upstream: `deinit` (segments.zig:45) — the join, handoff and
    /// side of every segment, then the fork; the Drop runs the same
    /// unwind at scope exit.
    pub fn deinit(&mut self) {
        for j in 0..self.n - 1 {
            self.join[j] = None;
            self.handoff[j] = None;
            self.side[j] = None;
        }
        self.fork = None;
        self.n = 1;
    }

    /// The stream segment k runs on beside the caller's. Upstream:
    /// `stream` (segments.zig:56).
    pub fn stream<'s>(&'s self, caller: &'s Stream<'d>, k: usize) -> &'s Stream<'s> {
        if k == 0 {
            caller
        } else {
            self.side[k - 1].as_ref().expect("the side stream")
        }
    }

    /// One call: `parts` segments through core/segments.zig's hooks on
    /// `fam`, forked from and joined back to `caller`. Upstream: `run`
    /// (segments.zig:61).
    pub fn run<F: Fam<E>, E: From<Error>>(
        &self,
        caller: &Stream<'d>,
        parts: usize,
        layers: usize,
        fam: &mut F,
    ) -> Result<(), E> {
        if parts == 0 || parts > self.n {
            return Err(E::from(Error::Segments));
        }
        // Upstream fills a fixed [MAX]Lane and slices; the port's
        // borrowed lanes live in a small Vec
        let lanes: Vec<Lane> = (0..parts)
            .map(|k| Lane {
                k,
                s: self.stream(caller, k),
            })
            .collect();
        let ops = RealOps { r: self, lanes: &lanes };
        stage(&ops, fam, &lanes, layers)
    }
}

impl Drop for Runner<'_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// The events between streams: lanes[0] is the caller's stream.
/// Upstream: `Ops` (segments.zig:71).
struct RealOps<'r, 'd> {
    r: &'r Runner<'d>,
    lanes: &'r [Lane<'r>],
}

impl RealOps<'_, '_> {
    fn fork(&self) -> Result<(), Error> {
        self.r
            .fork
            .as_ref()
            .expect("the fork event")
            .record(self.lanes[0].s)?;
        for l in &self.lanes[1..] {
            l.s.wait(self.r.fork.as_ref().expect("the fork event"))?;
        }
        Ok(())
    }
    fn signal(&self, k: usize) -> Result<(), Error> {
        self.r.handoff[k]
            .as_ref()
            .expect("the handoff event")
            .record(self.lanes[k].s)
    }
    fn wait_for(&self, k: usize) -> Result<(), Error> {
        self.lanes[k]
            .s
            .wait(self.r.handoff[k - 1].as_ref().expect("the handoff event"))
    }
    fn join(&self) -> Result<(), Error> {
        for (j, l) in self.lanes[1..].iter().enumerate() {
            self.r.join[j]
                .as_ref()
                .expect("the join event")
                .record(l.s)?;
            self.lanes[0]
                .s
                .wait(self.r.join[j].as_ref().expect("the join event"))?;
        }
        Ok(())
    }
    /// Blocks until every segment's queued work has ended (the error
    /// path, before buffers are reused).
    fn drain(&self) {
        for l in self.lanes {
            let _ = l.s.synchronize();
        }
    }
}

/// The events between streams, real or simulated: the ops `stage`
/// places. Upstream: the `ops` parameter of `stage` (segments.zig:98).
pub trait Ops {
    fn fork(&self) -> Result<(), Error>;
    fn signal(&self, k: usize) -> Result<(), Error>;
    fn wait_for(&self, k: usize) -> Result<(), Error>;
    fn join(&self) -> Result<(), Error>;
    fn drain(&self);
}

/// The family's segment hooks `stage` drives: the begin/pre/mixer/
/// post/finish set core/segments.zig's `drive` calls. Upstream: the
/// `fam` parameter of `stage` (segments.zig:98).
pub trait Fam<E = Error> {
    fn begin(&mut self, lane: &Lane) -> Result<(), E>;
    fn wait(&mut self, i: usize) -> Wait;
    fn handoff(&mut self, lane: &Lane, i: usize) -> Result<(), E>;
    fn pre(&mut self, lane: &Lane, i: usize) -> Result<(), E>;
    fn mixer(&mut self, lane: &Lane, i: usize) -> Result<(), E>;
    fn post(&mut self, lane: &Lane, i: usize) -> Result<(), E>;
    fn finish(&mut self, lane: &Lane) -> Result<(), E>;
}

impl Ops for RealOps<'_, '_> {
    fn fork(&self) -> Result<(), Error> {
        RealOps::fork(self)
    }
    fn signal(&self, k: usize) -> Result<(), Error> {
        RealOps::signal(self, k)
    }
    fn wait_for(&self, k: usize) -> Result<(), Error> {
        RealOps::wait_for(self, k)
    }
    fn join(&self) -> Result<(), Error> {
        RealOps::join(self)
    }
    fn drain(&self) {
        RealOps::drain(self)
    }
}

/// The schedule over `lanes`, with `ops` placing the fork, handoff and
/// join events (real streams or the test's). Upstream: `stage`
/// (segments.zig:98) — the X adapter drives `stagger.drive`; one
/// handoff event a pair serves every layer: a CUDA wait takes the
/// latest record, and drive queues it in order.
fn stage<O: Ops, F: Fam<E>, E>(ops: &O, fam: &mut F, lanes: &[Lane], layers: usize) -> Result<(), E>
where
    E: From<Error>,
{
    struct X<'x, O: Ops, F: Fam<E>, E> {
        ops: &'x O,
        fam: &'x mut F,
        lanes: &'x [Lane<'x>],
        _e: std::marker::PhantomData<E>,
    }
    impl<O: Ops, F: Fam<E>, E: From<Error>> stagger::Hooks for X<'_, O, F, E> {
        type Error = E;
        fn begin(&mut self, k: usize) -> Result<(), E> {
            self.fam.begin(&self.lanes[k])
        }
        fn wait(&mut self, i: usize) -> Wait {
            self.fam.wait(i)
        }
        fn wait_for(&mut self, k: usize, i: usize) -> Result<(), E> {
            self.ops.wait_for(k).map_err(|x| E::from(x))?;
            self.fam.handoff(&self.lanes[k], i)
        }
        fn signal(&mut self, k: usize, _: usize) -> Result<(), E> {
            self.ops.signal(k).map_err(|x| E::from(x))
        }
        fn pre(&mut self, k: usize, i: usize) -> Result<(), E> {
            self.fam.pre(&self.lanes[k], i)
        }
        fn mixer(&mut self, k: usize, i: usize) -> Result<(), E> {
            self.fam.mixer(&self.lanes[k], i)
        }
        fn post(&mut self, k: usize, i: usize) -> Result<(), E> {
            self.fam.post(&self.lanes[k], i)
        }
        fn finish(&mut self, k: usize) -> Result<(), E> {
            self.fam.finish(&self.lanes[k])
        }
    }
    let n = lanes.len();
    if n > 1 {
        ops.fork()?;
    }
    let mut x = X {
        ops,
        fam,
        lanes,
        _e: std::marker::PhantomData,
    };
    if let Err(err) = stagger::drive(n, layers, &mut x) {
        // join what was queued and let it end before the buffers are
        // reused
        if n > 1 {
            let _ = ops.join();
        }
        ops.drain();
        return Err(err);
    }
    if n > 1 {
        ops.join()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny xorshift for the test's random stream orders (upstream
    /// uses its own DefaultPrng; the test holds for every order, so
    /// only coverage differs).
    struct Xorshift(u64);
    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// CUDA's ordering in the tests (segments.zig:143's Sim): a record
    /// marks its stream's position; a wait takes the event's latest
    /// record when queued.
    struct Sim {
        ops: [Vec<Op>; MAX],
        latest: [Option<[usize; 2]>; 2 * MAX],
    }

    #[derive(Clone, Copy)]
    #[allow(dead_code)] // the record payload is never read back: the latest-record position lives in `latest`, as upstream's
    enum Op {
        Work(Work),
        Record(usize),
        Wait(Option<[usize; 2]>),
    }

    #[derive(Clone, Copy)]
    struct Work {
        call: usize,
        k: usize,
        kind: Kind,
        i: usize,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Before,
        Begin,
        Pre,
        Mixer,
        Post,
        Finish,
        After,
    }

    impl Sim {
        fn new() -> Sim {
            Sim {
                ops: [const { Vec::new() }; MAX],
                latest: [None; 2 * MAX],
            }
        }
        fn put(&mut self, stream: usize, op: Op) {
            self.ops[stream].push(op);
        }
        fn record(&mut self, ev: usize, stream: usize) {
            self.put(stream, Op::Record(ev));
            self.latest[ev] = Some([stream, self.ops[stream].len()]);
        }
        fn wait_on(&mut self, stream: usize, ev: usize) {
            self.put(stream, Op::Wait(self.latest[ev]));
        }

        /// Runs the streams one op at a time in an order `seed` picks,
        /// each work op through `check`; false on a deadlock.
        #[allow(clippy::needless_range_loop)] // upstream's own range loop over the streams
        fn execute(&self, n: usize, seed: u64, check: &mut Check) -> Result<bool, String> {
            let mut prng = Xorshift(seed | 1);
            let mut pc = [0usize; MAX];
            loop {
                let mut ready = [0usize; MAX];
                let mut m = 0;
                for k in 0..n {
                    if pc[k] == self.ops[k].len() {
                        continue;
                    }
                    let op = self.ops[k][pc[k]];
                    if let Op::Wait(Some(w)) = op {
                        if pc[w[0]] < w[1] {
                            continue;
                        }
                    }
                    ready[m] = k;
                    m += 1;
                }
                if m == 0 {
                    break;
                }
                let k = ready[prng.below(m)];
                let op = self.ops[k][pc[k]];
                if let Op::Work(w) = op {
                    check.see(w)?;
                }
                pc[k] += 1;
            }
            for k in 0..n {
                if pc[k] < self.ops[k].len() {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }

    /// The simulated runner's events: fork 0, handoff 1 + k, join MAX +
    /// k. Upstream: `SimOps` (segments.zig:190).
    #[derive(Clone, Copy)]
    struct SimOps {
        sim: *mut Sim,
        n: usize,
    }

    impl Ops for SimOps {
        fn fork(&self) -> Result<(), Error> {
            // SAFETY: the test's own simulator, alive through the call.
            let sim = unsafe { &mut *self.sim };
            sim.record(0, 0);
            for k in 1..self.n {
                sim.wait_on(k, 0);
            }
            Ok(())
        }
        fn signal(&self, k: usize) -> Result<(), Error> {
            // SAFETY: the test's own simulator, alive through the call.
            unsafe { &mut *self.sim }.record(1 + k, k);
            Ok(())
        }
        fn wait_for(&self, k: usize) -> Result<(), Error> {
            // SAFETY: the test's own simulator, alive through the call.
            unsafe { &mut *self.sim }.wait_on(k, 1 + (k - 1));
            Ok(())
        }
        fn join(&self) -> Result<(), Error> {
            // SAFETY: the test's own simulator, alive through the call.
            let sim = unsafe { &mut *self.sim };
            for k in 1..self.n {
                sim.record(MAX + k, k);
                sim.wait_on(0, MAX + k);
            }
            Ok(())
        }
        fn drain(&self) {}
    }

    /// A family that queues one work op per hook on its segment's
    /// stream. Upstream: `SimFam` (segments.zig:214).
    struct SimFam {
        sim: *mut Sim,
        call: usize,
        early: usize,
    }

    impl SimFam {
        fn mark(&mut self, k: usize, kind: Kind, i: usize) {
            // SAFETY: the test's own simulator, alive through the call.
            unsafe { &mut *self.sim }.put(
                k,
                Op::Work(Work {
                    call: self.call,
                    k,
                    kind,
                    i,
                }),
            );
        }
    }

    impl Fam for SimFam {
        fn begin(&mut self, l: &Lane) -> Result<(), Error> {
            self.mark(l.k, Kind::Begin, 0);
            Ok(())
        }
        fn wait(&mut self, i: usize) -> Wait {
            if i == self.early {
                Wait::Pre
            } else {
                Wait::Mixer
            }
        }
        fn handoff(&mut self, _: &Lane, _: usize) -> Result<(), Error> {
            Ok(())
        }
        fn pre(&mut self, l: &Lane, i: usize) -> Result<(), Error> {
            self.mark(l.k, Kind::Pre, i);
            Ok(())
        }
        fn mixer(&mut self, l: &Lane, i: usize) -> Result<(), Error> {
            self.mark(l.k, Kind::Mixer, i);
            Ok(())
        }
        fn post(&mut self, l: &Lane, i: usize) -> Result<(), Error> {
            self.mark(l.k, Kind::Post, i);
            Ok(())
        }
        fn finish(&mut self, l: &Lane) -> Result<(), Error> {
            self.mark(l.k, Kind::Finish, 0);
            Ok(())
        }
    }

    /// What must hold whatever order the streams run in. Upstream:
    /// `Check` (segments.zig:244).
    struct Check {
        n: usize,
        early: usize,
        before: [bool; 2],
        /// Mixers run, per call and segment.
        mixed: [[usize; MAX]; 2],
        ended: [usize; 2],
    }

    impl Check {
        fn see(&mut self, w: Work) -> Result<(), String> {
            match w.kind {
                Kind::Before => self.before[w.call] = true,
                Kind::Begin => {
                    // side segments start after the caller's earlier
                    // work, and after the last call ended
                    if !self.before[w.call] {
                        return Err("begin before the caller's earlier work".into());
                    }
                    if w.call > 0 && self.ended[w.call - 1] != self.n {
                        return Err("begin before the last call ended".into());
                    }
                }
                Kind::Pre => {
                    if w.k > 0 && w.i == self.early && self.mixed[w.call][w.k - 1] <= w.i {
                        return Err(format!(
                            "pre of segment {} at layer {} before the segment before's mixer",
                            w.k, w.i
                        ));
                    }
                }
                Kind::Mixer => {
                    if w.k > 0 && self.mixed[w.call][w.k - 1] <= w.i {
                        return Err(format!(
                            "mixer of segment {} at layer {} before the segment before's mixer",
                            w.k, w.i
                        ));
                    }
                    self.mixed[w.call][w.k] = w.i + 1;
                }
                Kind::Post => {}
                Kind::Finish => self.ended[w.call] += 1,
                Kind::After => {
                    // the caller's later work sees every segment
                    if self.ended[w.call] != self.n {
                        return Err("the caller's later work before every segment ended".into());
                    }
                }
            }
            Ok(())
        }
    }

    /// Upstream's `the CUDA schedule keeps every dependency with reused
    /// events` (segments.zig:272) — two calls with reused events, in
    /// many random stream orders: every dependency in Check holds and
    /// no stream deadlocks.
    #[test]
    fn the_cuda_schedule_keeps_every_dependency_with_reused_events() {
        let layers = 6;
        for n in 1..=MAX {
            let mut sim = Sim::new();
            let mut fam = SimFam {
                sim: &mut sim,
                call: 0,
                early: 1,
            };
            let dummy = crate::driver::stub::stub_stream();
            let lanes: Vec<Lane> = (0..n)
                .map(|k| Lane { k, s: &dummy })
                .collect();
            for call in 0..2 {
                fam.call = call;
                sim.put(
                    0,
                    Op::Work(Work {
                        call,
                        k: 0,
                        kind: Kind::Before,
                        i: 0,
                    }),
                );
                stage(&SimOps { sim: &mut sim, n }, &mut fam, &lanes, layers).unwrap();
                sim.put(
                    0,
                    Op::Work(Work {
                        call,
                        k: 0,
                        kind: Kind::After,
                        i: 0,
                    }),
                );
            }
            for seed in 0..200u64 {
                let mut check = Check {
                    n,
                    early: 1,
                    before: [false; 2],
                    mixed: [[0; MAX]; 2],
                    ended: [0; 2],
                };
                assert!(
                    sim.execute(n, seed, &mut check).unwrap(),
                    "no stream deadlocked (n {n}, seed {seed})"
                );
                assert_eq!(n, check.ended[1]);
            }
        }
    }

    /// Upstream's `the simulated check catches a missing handoff`
    /// (segments.zig:294) — the same streams without the handoff waits
    /// let some order run a mixer before the segment before's: the
    /// check bites.
    #[test]
    fn the_simulated_check_catches_a_missing_handoff() {
        /// NoWait: the inner ops without the handoff waits.
        struct NoWait {
            inner: SimOps,
        }
        impl Ops for NoWait {
            fn fork(&self) -> Result<(), Error> {
                self.inner.fork()
            }
            fn signal(&self, k: usize) -> Result<(), Error> {
                self.inner.signal(k)
            }
            fn wait_for(&self, _: usize) -> Result<(), Error> {
                Ok(())
            }
            fn join(&self) -> Result<(), Error> {
                self.inner.join()
            }
            fn drain(&self) {}
        }
        let mut sim = Sim::new();
        let mut fam = SimFam {
            sim: &mut sim,
            call: 0,
            early: 1,
        };
        let dummy = crate::driver::stub::stub_stream();
        let lanes: Vec<Lane> = vec![Lane { k: 0, s: &dummy }, Lane { k: 1, s: &dummy }];
        sim.put(
            0,
            Op::Work(Work {
                call: 0,
                k: 0,
                kind: Kind::Before,
                i: 0,
            }),
        );
        stage(
            &NoWait {
                inner: SimOps { sim: &mut sim, n: 2 },
            },
            &mut fam,
            &lanes,
            4,
        )
        .unwrap();
        let mut caught = false;
        for seed in 0..200u64 {
            let mut check = Check {
                n: 2,
                early: 1,
                before: [false; 2],
                mixed: [[0; MAX]; 2],
                ended: [0; 2],
            };
            if sim.execute(2, seed, &mut check).is_err() {
                caught = true;
                break;
            }
        }
        assert!(caught);
    }
}
