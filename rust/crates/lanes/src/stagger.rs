//! Staggered prompt segments, backend-neutral: calls, segment rows, and
//! the hook order core/ and cuda/segments.zig run.
//!
//! Port of `zig/src/core/stagger.zig` (180 lines): the row arithmetic
//! (near-equal splits, the first ones longer), the call planner, the
//! chunked planner, and the `drive` hook order — begins; per layer, per
//! segment in row order: pre, mixer, post, k waiting on k-1's
//! post-mixer signal. Upstream's three tests ported whole (the third
//! through a queue simulation whose random orders the port drives with
//! its own xorshift — the test holds for every order, so the generator
//! is the port's, the contract upstream's).

/// Segments one chunk can take.
pub const MAX: usize = 4;

/// Where layer i waits for the segment before: ahead of pre (it reads
/// what that one wrote, a conv tail) or of the mixer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Wait {
    Pre,
    Mixer,
}

/// Rows of segment k when `n` rows split `parts` ways (near-equal, the
/// first ones longer). Upstream: `rows` (stagger.zig:14).
pub fn rows(n: usize, parts: usize, k: usize) -> usize {
    n / parts + usize::from(k < n % parts)
}

/// The first row of segment k. Upstream: `start` (stagger.zig:18).
pub fn start(n: usize, parts: usize, k: usize) -> usize {
    k * (n / parts) + k.min(n % parts)
}

/// Upstream: `Call` (stagger.zig:21).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Call {
    pub rows: usize,
    pub parts: usize,
}

/// A prompt's next call: `left` rows evenly over the fewest calls of two
/// segments of up to `step`, each `min_rows` or more. Upstream: `next`
/// (stagger.zig:25).
pub fn next(left: usize, step: usize, min_rows: usize) -> Call {
    let calls = left.div_ceil(2 * step).max(1);
    let r = left.div_ceil(calls);
    if r < 2 * min_rows {
        return Call { rows: step.min(left), parts: 1 };
    }
    Call { rows: r, parts: 2 }
}

/// A prompt's next call of up to `parts` whole serial chunks of `step`
/// rows: each segment is the chunk at its place. Upstream: `chunked`
/// (stagger.zig:32).
pub fn chunked(left: usize, step: usize, parts: usize) -> Call {
    let r = left.min(parts * step);
    Call { rows: r, parts: r.div_ceil(step).max(1) }
}

/// Rows of segment k of a chunked call of `n` rows. Upstream:
/// `chunkRows` (stagger.zig:38).
pub fn chunk_rows(n: usize, step: usize, k: usize) -> usize {
    step.min(n - k * step)
}

/// The hook order: begins; per layer, per segment in row order: pre,
/// mixer, post, k waiting on k-1's post-mixer signal. Upstream: `drive`
/// (stagger.zig:42). The hooks come as a trait so both the core and
/// the CUDA schedule run the same order.
pub trait Hooks {
    type Error;
    fn begin(&mut self, k: usize) -> Result<(), Self::Error>;
    fn wait(&mut self, i: usize) -> Wait;
    fn wait_for(&mut self, k: usize, i: usize) -> Result<(), Self::Error>;
    fn pre(&mut self, k: usize, i: usize) -> Result<(), Self::Error>;
    fn mixer(&mut self, k: usize, i: usize) -> Result<(), Self::Error>;
    fn signal(&mut self, k: usize, i: usize) -> Result<(), Self::Error>;
    fn post(&mut self, k: usize, i: usize) -> Result<(), Self::Error>;
    fn finish(&mut self, k: usize) -> Result<(), Self::Error>;
}

/// Upstream: `drive` (stagger.zig:42) — k-1's hooks queue before k's
/// (host counters move in row order; a reused event's wait follows its
/// signal).
pub fn drive<X: Hooks>(n: usize, layers: usize, x: &mut X) -> Result<(), X::Error> {
    for k in 0..n {
        x.begin(k)?;
    }
    for i in 0..layers {
        let w = x.wait(i);
        for k in 0..n {
            if k > 0 && w == Wait::Pre {
                x.wait_for(k, i)?;
            }
            x.pre(k, i)?;
            if k > 0 && w == Wait::Mixer {
                x.wait_for(k, i)?;
            }
            x.mixer(k, i)?;
            if k + 1 < n {
                x.signal(k, i)?;
            }
            x.post(k, i)?;
        }
    }
    for k in 0..n {
        x.finish(k)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `rows, starts and calls` (stagger.zig:58).
    #[test]
    fn rows_starts_and_calls() {
        let mut at = 0;
        for k in 0..3 {
            assert_eq!(start(10, 3, k), at);
            at += rows(10, 3, k);
        }
        assert_eq!(at, 10);
        assert_eq!(next(40000, 8192, 4096), Call { rows: 13334, parts: 2 }); // three even calls
        assert_eq!(next(20000, 8192, 4096), Call { rows: 10000, parts: 2 });
        assert_eq!(next(8625, 8192, 4096), Call { rows: 8625, parts: 2 });
        assert_eq!(next(4096, 8192, 4096), Call { rows: 4096, parts: 1 });
        assert_eq!(next(9000, 8192, 8192), Call { rows: 8192, parts: 1 });
        assert_eq!(next(0, 8192, 4096), Call { rows: 0, parts: 1 });
        let mut left = 70001usize; // every call of a long prompt keeps both segments at min_rows or more
        while left > 0 {
            let c = next(left, 8192, 4096);
            assert!(
                c.rows <= 2 * 8192 && (c.parts == 1 || c.rows >= 2 * 4096)
            );
            left -= c.rows;
        }
    }

    /// Upstream's `chunked calls cut a prompt exactly where serial chunks
    /// do` (stagger.zig:71).
    #[test]
    fn chunked_calls_cut_a_prompt_exactly_where_serial_chunks_do() {
        assert_eq!(chunked(8203, 2048, 2), Call { rows: 4096, parts: 2 });
        assert_eq!(chunked(11, 2048, 4), Call { rows: 11, parts: 1 });
        assert_eq!(chunked(2059, 2048, 3), Call { rows: 2059, parts: 2 });
        assert_eq!(chunked(2048, 2048, 2), Call { rows: 2048, parts: 1 });
        assert_eq!(chunk_rows(2059, 2048, 1), 11);
        for len in [1, 2047, 2048, 2049, 8203, 32789, 70001] {
            for parts in 1..=MAX {
                let mut pos = 0; // every segment is the serial chunk at its place: start on the grid, full but the last
                while pos < len {
                    let c = chunked(len - pos, 2048, parts);
                    assert!(c.parts >= 1 && c.parts <= parts && c.rows > 0);
                    let mut sum = 0;
                    for k in 0..c.parts {
                        let r = chunk_rows(c.rows, 2048, k);
                        assert!(
                            (pos + sum) % 2048 == 0 && r > 0 && r <= 2048,
                            "segment {k} of {c:?}: r {r} at {}",
                            pos + sum
                        );
                        assert!(r == 2048 || pos + sum + r == len);
                        sum += r;
                    }
                    assert_eq!(c.rows, sum);
                    pos += c.rows;
                }
                assert_eq!(len, pos);
            }
        }
    }

    /// The schedule on simulated queues (upstream's third test,
    /// stagger.zig:120): a wait blocks until the segment before signalled
    /// its layer; each hook after its waits.
    struct Recorder {
        steps: [Vec<Step>; MAX],
        early: usize,
    }

    #[derive(Clone, Copy)]
    struct Step {
        kind: Kind,
        i: usize,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Begin,
        Wait,
        Pre,
        Mixer,
        Signal,
        Post,
        Finish,
    }

    impl Recorder {
        fn add(&mut self, k: usize, kind: Kind, i: usize) {
            self.steps[k].push(Step { kind, i });
        }
    }

    impl Hooks for Recorder {
        type Error = ();
        fn begin(&mut self, k: usize) -> Result<(), ()> {
            self.add(k, Kind::Begin, 0);
            Ok(())
        }
        fn wait(&mut self, i: usize) -> Wait {
            if i == self.early { Wait::Pre } else { Wait::Mixer }
        }
        fn wait_for(&mut self, k: usize, i: usize) -> Result<(), ()> {
            self.add(k, Kind::Wait, i);
            Ok(())
        }
        fn pre(&mut self, k: usize, i: usize) -> Result<(), ()> {
            self.add(k, Kind::Pre, i);
            Ok(())
        }
        fn mixer(&mut self, k: usize, i: usize) -> Result<(), ()> {
            self.add(k, Kind::Mixer, i);
            Ok(())
        }
        fn signal(&mut self, k: usize, i: usize) -> Result<(), ()> {
            self.add(k, Kind::Signal, i);
            Ok(())
        }
        fn post(&mut self, k: usize, i: usize) -> Result<(), ()> {
            self.add(k, Kind::Post, i);
            Ok(())
        }
        fn finish(&mut self, k: usize) -> Result<(), ()> {
            self.add(k, Kind::Finish, 0);
            Ok(())
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // upstream's own range walk over the queues
    fn the_schedule_runs_to_the_end_on_queues_and_keeps_each_layer_s_dependencies() {
        for n in 1..=MAX {
            let mut rec = Recorder {
                steps: [const { Vec::new() }; MAX],
                early: 1,
            };
            drive(n, 4, &mut rec).unwrap();
            // the queues: a wait for layer i passes once segment k-1
            // signalled layer i; nothing else blocks
            let mut pc = [0usize; MAX];
            let mut signalled = [Option::<usize>::None; MAX];
            let mut mixed = [0usize; MAX]; // layers whose mixer has run
            let mut moved = true;
            while moved {
                moved = false;
                for k in 0..n {
                    while pc[k] < rec.steps[k].len() {
                        let s = rec.steps[k][pc[k]];
                        if s.kind == Kind::Wait
                            && (signalled[k - 1].is_none() || signalled[k - 1] < Some(s.i))
                        {
                            break;
                        }
                        match s.kind {
                            Kind::Signal => signalled[k] = Some(s.i),
                            Kind::Mixer => {
                                if k > 0 {
                                    assert!(mixed[k - 1] > s.i, "the segment before's mixer is done");
                                }
                                mixed[k] = s.i + 1;
                            }
                            Kind::Pre if k > 0 && s.i == rec.early => {
                                assert!(mixed[k - 1] > s.i);
                            }
                            _ => {}
                        }
                        pc[k] += 1;
                        moved = true;
                    }
                }
            }
            for k in 0..n {
                assert_eq!(rec.steps[k].len(), pc[k], "no queue left waiting");
            }
            for k in 0..n {
                // begin, finish, three parts a layer, a wait a layer past
                // the first, a signal before the last
                let want = 2 + 3 * 4 + if k > 0 { 4 } else { 0 } + if k + 1 < n { 4 } else { 0 };
                assert_eq!(rec.steps[k].len(), want);
            }
        }
    }
}
