//! The lane round loop (Python FamilyRounds and SharedRounds): every decision and
//! cost update in Python's order.
//!
//! Port of `zig/src/core/lanes/engine.zig`. Stream pointers follow the crate's
//! invariants (lib.rs); the arena's contract is on `Engine::arena`.

use crate::backend::{Backend, Clock, DraftRequest, Feed, Verified, Window};
use crate::config::Config;
use crate::depth::{self, Rule};
use crate::events::{f, Log, Value};
use crate::fill;
use crate::plan_lanes::{self, Costs, Grafts};
use crate::shape;
use crate::stream::{self, Held, Mode, Stream};
use crate::windows::{self, Kind, Plan};
use crate::{Error, Mark};
use bumpalo::Bump;
use tensorfold_engine::LogprobRow;

/// Upstream: `engine.zig:23-25` — the round's outcome records.
struct Outcome {
    #[allow(dead_code)]
    got: &'static [u32],
    path: &'static [u32],
    cut: Option<usize>,
    follow: &'static [u32],
}

#[derive(Clone, Copy)]
struct Result_ {
    rows: u32,
    keep: u32,
}

pub(crate) struct Unread {
    pub(crate) event: usize,
    pub(crate) handle: u64,
}

/// Upstream: `engine.zig:27  pub const Engine`.
pub struct Engine<'a> {
    pub cfg: &'a Config,
    pub backend: &'a mut dyn Backend,
    pub clock: &'a mut dyn Clock,
    /// The decision log, when the recorder is on (null in production).
    pub log: Option<Log>,
    pub rule: Rule,
    /// Admitted streams in admission order (Python `_live`).
    pub live: Vec<*mut Stream<'a>>,
    pub drafted: u64,
    pub accepted: u64,
    pub shared_rounds: u64,
    /// This step has one live stream: its copies may take their ramp.
    pub alone: bool,
    pub steps: u64,
    /// A round's temporaries.
    arena: Bump,
    /// Queued draws the log reads at the step's end (the trail resolves them).
    pub(crate) unread: Vec<Unread>,
}

// Safety: the host moves the whole engine across exactly one thread boundary
// (its construction to the engine thread, inside the host); from there the one
// engine thread touches it alone, under the crate's invariants (lib.rs). The
// status and submit paths reach only the host's own shared state, never the
// core.
unsafe impl Send for Engine<'_> {}

impl<'a> Engine<'a> {
    /// Upstream: `init` (engine.zig:43).
    pub fn init(
        cfg: &'a Config,
        backend: &'a mut dyn Backend,
        clock: &'a mut dyn Clock,
    ) -> Engine<'a> {
        let rule = Rule::new(
            cfg.k,
            &cfg.depth_prior,
            cfg.most_drafts,
            &cfg.family_costs,
            &cfg.shared_costs,
            cfg.mtp_step_ms,
            cfg.plain_guard,
            cfg.node_probabilities,
            cfg.batch_rows,
        );
        Engine {
            cfg,
            backend,
            clock,
            log: None,
            rule,
            live: Vec::new(),
            drafted: 0,
            accepted: 0,
            shared_rounds: 0,
            alone: true,
            steps: 0,
            arena: Bump::new(),
            unread: Vec::new(),
        }
    }

    /// The round's arena. Upstream's arena is a separate allocator whose product
    /// slices outlive any one borrow of the engine; this accessor gives the port
    /// the same shape.
    ///
    /// Safety: sound under the crate's invariants — one thread owns the engine,
    /// the arena resets only between rounds (`reset` below), and every slice the
    /// arena gives out is dead before the round that allocated it ends (the
    /// round's temporaries never escape `step`/`add_stream`).
    pub(crate) fn arena(&self) -> &'static Bump {
        unsafe { &*(&self.arena as *const Bump) }
    }

    /// Upstream: `_ = e.arena.reset(.retain_capacity)` — the round's temporaries
    /// die with the round.
    fn reset_arena(&mut self) {
        self.arena.reset();
    }

    /// Streams holding or about to hold a row (paused ones included).
    /// Upstream: `activeCount`.
    pub fn active_count(&self) -> usize {
        self.live
            .iter()
            .filter(|&&s| unsafe { !(*s).finished })
            .count()
    }

    /// Prefill a stream, draw its first token and ask for its first drafts; it
    /// takes part from the next round. Upstream: `addStream` (engine.zig:80).
    ///
    /// Safety: `s` per the crate invariants; the engine adopts it.
    ///
    /// # Safety
    ///
    /// `s` is a live, boxed stream this engine owns from its admission until
    /// its release (the crate's invariants, lib.rs).
    pub unsafe fn add_stream(&mut self, s: *mut Stream<'a>) -> Result<(), Error> {
        self.reset_arena();
        // Safety: the stream pointer, under the crate invariants.
        let stream = unsafe { &mut *s };
        crate::trail::event(
            self,
            &[f("ev", Value::Str("add")), f("stream", Value::Str(stream.id))],
        )?;
        match self.backend.prefill(s) {
            Ok(()) => {}
            Err(Error::Cancelled) => {
                // The host finishes a cancelled stream without the core.
                self.backend.release(s);
                return Err(Error::Cancelled);
            }
            Err(e) => return Err(e),
        }
        if stream.is_cancelled() {
            // Cancelled in its last chunk: no first token.
            self.backend.release(s);
            return Err(Error::Cancelled);
        }
        stream.context.truncate(stream.prompt_len);
        stream.rows.clear();
        stream.pending = None;
        stream.cache_len = stream.prompt_len as u64;
        let position: u64 = stream.prompt_len as u64;
        let drawn = self.backend.first(s, position)?;
        let mut feed = Feed::Handle(drawn);
        if let Some(t) = self.forced_next(s)? {
            feed = Feed::Value(t);
        }
        let mut asked: Option<u32> = None;
        if self.cfg.family_mtp && stream.drafts {
            // The head reads the prompt's last row and the first token, and drafts
            // the one after it.
            // Safety: the stream pointer.
            let who = unsafe { windows::who(s) };
            let d = self.rule.depth(who)? as u32;
            asked = Some(d);
            let mut request = DraftRequest::new(s, &[], stream.prompt_len as u64, position + 1, d);
            // Upstream: `.first = feed` — the head reads the drawn first token.
            request.first = Some(feed);
            self.backend.draft(&[request])?;
            stream.drop_held();
            stream.next = Some(Held { count: d, ..Held::default() });
            if let Some(held) = self.backend.tree(s)? {
                // A tree head's first drafts as host tokens, as every later
                // round's.
                stream.next = Some(held);
            }
        } else if self.cfg.pipelined && stream.logprobs.is_none() {
            self.queue_next(s, feed)?;
        }
        let value = self.read_feed(feed)?;
        if let Some(d) = asked {
            crate::trail::event(
                self,
                &[
                    f("ev", Value::Str("draft")),
                    f("stream", Value::Str(stream.id)),
                    f("depth", Value::Int(d as i64)),
                    f("position", Value::Int((position + 1) as i64)),
                    f("follow", Value::U32s(&[value])),
                    f("rows", Value::Null),
                ],
            )?;
        }
        if self.log.is_some() {
            let first = match feed {
                Feed::Handle(_) => value,
                Feed::Value(_) => self.backend.read(drawn)?,
            };
            crate::trail::event(
                self,
                &[
                    f("ev", Value::Str("first")),
                    f("stream", Value::Str(stream.id)),
                    f("position", Value::Int(position as i64)),
                    f("drawn", Value::Int(first as i64)),
                    f("token", Value::Int(value as i64)),
                ],
            )?;
        }
        let mut first_row = [LogprobRow::default(); 1];
        if stream.logprobs.is_some() {
            first_row[0] = self.backend.first_row(s)?.for_token(value);
        }
        let rows: &[LogprobRow] = if stream.logprobs.is_some() { &first_row } else { &[] };
        stream.commit(&[value], rows)?;
        stream.pending = Some(value);
        crate::trail::resolve(self)?;
        if stream.finished {
            crate::trail::finish(self, s)?;
            self.release(s);
            return Ok(());
        }
        self.live.push(s);
        Ok(())
    }

    /// A stream another driver prefilled and drew `token` for: drafts asked, the
    /// token committed, rounds from here. Upstream: `adoptPrefilled` (engine.zig:133).
    ///
    /// Safety: as `add_stream`.
    ///
    /// # Safety
    ///
    /// As `add_stream`.
    pub unsafe fn adopt_prefilled(&mut self, s: *mut Stream<'a>, token: u32) -> Result<(), Error> {
        self.reset_arena();
        // Safety: the stream pointer.
        let stream = unsafe { &mut *s };
        stream.context.truncate(stream.prompt_len);
        stream.pending = None;
        stream.cache_len = stream.prompt_len as u64;
        if self.cfg.family_mtp && stream.drafts {
            // Safety: the stream pointer.
            let who = unsafe { windows::who(s) };
            let d = self.rule.depth(who)? as u32;
            let mut request =
                DraftRequest::new(s, &[], stream.prompt_len as u64, stream.prompt_len as u64 + 1, d);
            request.first = Some(Feed::Value(token));
            self.backend.draft(&[request])?;
            stream.drop_held();
            stream.next = Some(Held { count: d, ..Held::default() });
            if let Some(held) = self.backend.tree(s)? {
                stream.next = Some(held);
            }
        }
        if stream.logprobs.is_some() {
            // Another driver's prompt pass gives no first row.
            return Err(Error::LogprobsUnsupported);
        }
        stream.commit(&[token], &[])?;
        stream.pending = Some(token);
        if stream.finished {
            self.release(s);
            return Ok(());
        }
        self.live.push(s);
        Ok(())
    }

    /// Take over a stream another driver decoded so far: its cache settled,
    /// `pending` and `cache_len` set, no drafts held. Upstream: `adopt`.
    ///
    /// Safety: as `add_stream`.
    ///
    /// # Safety
    ///
    /// `s` is a live, boxed stream this engine owns from its admission until
    /// its release (the crate's invariants, lib.rs).
    pub unsafe fn adopt(&mut self, s: *mut Stream<'a>) -> Result<(), Error> {
        self.live.push(s);
        Ok(())
    }

    /// Release a cancelled stream between rounds: no pending draws or drafts.
    /// Upstream: `discard` (engine.zig:160).
    ///
    /// Safety: `s` must be one this engine adopted.
    ///
    /// # Safety
    ///
    /// As `add_stream`.
    pub unsafe fn discard(&mut self, s: *mut Stream<'a>) {
        // Safety: the stream pointer.
        unsafe {
            (*s).finished = true;
            (*s).reason = stream::Reason::Cancelled;
        }
        if let Some(i) = self.live.iter().position(|&x| x == s) {
            self.live.remove(i);
        }
        self.release(s);
    }

    /// One round for every live stream (Python `_family_step`).
    /// Upstream: `step` (engine.zig:173).
    pub fn step(&mut self) -> Result<(), Error> {
        self.reset_arena();
        crate::trail::event(
            self,
            &[f("ev", Value::Str("step")), f("index", Value::Int(self.steps as i64))],
        )?;
        self.steps += 1;
        let arena = self.arena();
        // Safety: every live pointer, under the crate invariants.
        let mut live: Vec<*mut Stream<'a>> = self
            .live
            .iter()
            .copied()
            .filter(|&s| unsafe { !(*s).finished && !(*s).paused })
            .collect();
        if live.len() > 1 && self.cfg.family_streams {
            // A shared round is synchronous: a stream that ran a step ahead lands
            // its queued token first.
            for &s in live.clone().iter() {
                // Safety: the stream pointer.
                if unsafe { (*s).inflight.is_some() } {
                    self.land_inflight(s)?;
                }
            }
            live.retain(|&s| unsafe { !(*s).finished });
        }
        self.alone = live.len() == 1;
        if live.len() > 1 && self.cfg.family_streams {
            let chosen = self.take_turns(&live)?;
            self.shared_rounds += 1;
            for &s in &chosen {
                // Safety: the stream pointer.
                unsafe {
                    (*s).served = Some(self.shared_rounds);
                }
            }
            self.round_streams(&chosen)?;
        } else {
            for &s in &live {
                // Safety: the stream pointer.
                let (windowed, logprobs, drafts) = unsafe {
                    let x = &*s;
                    (
                        (self.cfg.family_mtp && x.drafts)
                            || !self.cfg.pipelined
                            || x.logprobs.is_some(),
                        x.logprobs.is_some(),
                        x.drafts,
                    )
                };
                let r = if windowed {
                    self.family_round(s, None)?
                } else {
                    self.pipelined_round(s)?
                };
                // Safety: the stream pointer.
                unsafe {
                    let x = &mut *s;
                    x.min_rows = if x.min_rows == 0 {
                        r.rows
                    } else {
                        x.min_rows.min(r.rows)
                    };
                }
                let _ = (logprobs, drafts);
                let _ = arena;
            }
        }
        crate::trail::resolve(self)?;
        let mut kept: usize = 0;
        let live_now: Vec<*mut Stream<'a>> = self.live.clone();
        for &s in &live_now {
            // Safety: the stream pointer.
            if unsafe { (*s).finished } {
                crate::trail::finish(self, s)?;
                self.release(s);
                continue;
            }
            self.live[kept] = s;
            kept += 1;
        }
        self.live.truncate(kept);
        crate::trail::state(self)?;
        Ok(())
    }

    /// Least recently served streams within the stream and row limits, never
    /// cutting pending or forced rows. Upstream: `takeTurns` (engine.zig:225).
    fn take_turns(&mut self, live: &[*mut Stream<'a>]) -> Result<Vec<*mut Stream<'a>>, Error> {
        let arena = self.arena();
        // Safety: the live pointers.
        let served = |s: *mut Stream<'a>| -> i64 {
            unsafe { (*s).served.map(|v| v as i64).unwrap_or(-1) }
        };
        let mut order: Vec<usize> = (0..live.len()).collect();
        order.sort_by(|&x, &y| {
            let sx = served(live[x]);
            let sy = served(live[y]);
            sx.cmp(&sy).then(x.cmp(&y))
        });
        let mut chosen: Vec<usize> = Vec::new();
        let mut rows: u64 = 0;
        for i in order {
            // Safety: the stream pointer.
            let need: u64 = unsafe {
                let s = &*live[i];
                if s.drafts {
                    self.cfg.base_width.min(self.cfg.batch_rows).min(1 + s.force.len() as u32) as u64
                } else {
                    1
                }
            };
            if chosen.len() == self.cfg.batch_streams as usize
                || (!chosen.is_empty() && rows + need > self.cfg.batch_rows as u64)
            {
                break;
            }
            chosen.push(i);
            rows += need;
        }
        chosen.sort_unstable();
        let out: Vec<*mut Stream<'a>> = chosen.iter().map(|&i| live[i]).collect();
        let ids: Vec<Value<'_>> = out
            .iter()
            .map(|&s| Value::Str(unsafe { (*s).id }))
            .collect();
        crate::trail::event(
            self,
            &[
                f("ev", Value::Str("turns")),
                f("streams", Value::List(arena.alloc_slice_copy(&ids))),
            ],
        )?;
        Ok(out)
    }

    /// One stream's round: verify pending and drafted tokens together, keep drafts
    /// to the first mismatch. Upstream: `familyRound` (engine.zig:262).
    fn family_round(
        &mut self,
        s: *mut Stream<'a>,
        copied: Option<&'static [u32]>,
    ) -> Result<Result_, Error> {
        self.clock.start();
        // Safety: the stream pointer.
        let priced = unsafe { (*s).priced.take() };
        let mut plans = [self.plan_one(s, copied)?];
        let plan_kind = plans[0].kind;
        let plan_parents = plans[0].parents.is_some();
        let plan_lanes = plans[0].lanes.is_some();
        let plan_position = plans[0].position;
        if plan_kind == Kind::Head && self.cfg.node_probabilities {
            windows::allocate(self, &mut plans)?;
        }
        let plan = &plans[0];
        let early = self.cfg.family_mtp
            && unsafe { (*s).drafts }
            && plan_kind != Kind::Forced
            && self.cfg.speculate_early
            && !plan_parents;
        let window = windows::build(self, plan, early)?;
        let rows = window.rows();
        let arena = self.arena();
        let out_sampled: &'static mut [u32] = arena.alloc_slice_copy(&vec![0u32; rows]);
        let out_drafts: &'static mut [u32] = arena.alloc_slice_copy(&vec![0u32; rows - 1]);
        // Safety: the stream pointer.
        let wants_rows = unsafe { (*s).logprobs.is_some() };
        let out_rows: &'static mut [LogprobRow] =
            if wants_rows { arena.alloc_slice_copy(&vec![LogprobRow::default(); rows]) } else { &mut [] };
        let mut verified = [Verified {
            sampled: out_sampled,
            drafts: out_drafts,
            rows: out_rows,
        }];
        self.backend.verify(&[window], &mut verified)?;
        // The 'static muts move back out, so the round can read what the backend
        // wrote without borrowing the local.
        let [Verified { sampled: o_sampled, drafts: o_drafts, rows: o_rows }] = verified;
        let tokens = windows::tokens(
            self,
            &window,
            &Verified { sampled: o_sampled, drafts: o_drafts, rows: o_rows },
        );
        let parents = windows::row_parents(self, &window)?;
        let sampled: &'static [u32] = o_sampled;
        let rows_lp: &'static [LogprobRow] = o_rows;
        let positions = window.positions;
        let o = self.conclude(
            s,
            plan_kind,
            plan_parents,
            plan_lanes,
            plan_position,
            plan.forced,
            plan.branch_from,
            plan.tried,
            plan.lanes,
            sampled,
            rows_lp,
            tokens,
            parents,
            positions,
        )?;
        if o.path.len() < rows || plan_parents {
            self.backend.keep(&[window], &[o.path])?;
        }
        // Safety: the stream pointer.
        let held_levels = unsafe { (*s).held_levels };
        if self.cfg.family_mtp && unsafe { (*s).drafts } {
            if early && o.cut.is_none() {
                // Safety: the stream pointer.
                let who = unsafe { windows::who(s) };
                let probe = windows::probe(self);
                let d = self.rule.head_depth(&who, None, probe)? as u32;
                let mut request = DraftRequest::new(s, o.follow, plan_position, unsafe {
                    (*s).cache_len + 1
                }, d);
                request.early = true;
                request.rows = Some(o.path);
                self.ask_drafts(&mut [request])?;
            } else {
                if early {
                    self.backend.unspeculate(s)?;
                }
                self.draft_late(s, plan_position, o.follow, o.path, None)?;
            }
            if plan_kind == Kind::Head || (plan_kind == Kind::None && self.cfg.plain_guard) {
                let ms = self.clock.elapsed_ms(Mark::Cost);
                // Safety: the stream pointer.
                let first = unsafe { (*s).rounds == 1 };
                crate::trail::event(
                    self,
                    &[
                        f("ev", Value::Str("cost")),
                        f("stream", Value::Str(unsafe { (*s).id })),
                        f("drafts", Value::Int((rows - 1) as i64)),
                        f("init", Value::Bool(first)),
                        f("ms", crate::events::bits(ms)),
                    ],
                )?;
                // Safety: the stream pointer.
                let who = unsafe { windows::who(s) };
                self.rule
                    .observe_cost(rows as i64 - 1, ms, first, Some(who))?;
            }
        }
        // Safety: the stream pointer.
        let (graft_room, rounds) = unsafe { ((*s).graft_room, (*s).rounds) };
        if graft_room.is_some() && rounds > 1 {
            // The stream's round time beyond its window and head levels, for the
            // lane planner.
            if let Some(w) = self.cfg.lane_costs.get(rows as i64) {
                let over =
                    self.clock.elapsed_ms(Mark::Cost) - w - self.cfg.mtp_step_ms * held_levels as f64;
                // Safety: the stream pointer.
                unsafe {
                    let s = &mut *s;
                    let kept = if plan_lanes && plan_parents {
                        &mut s.tree_over
                    } else {
                        &mut s.round_over
                    };
                    *kept = match *kept {
                        Some(x) => Some(x + 0.25 * (over - x)),
                        None => Some(over),
                    };
                }
            }
            // The pick that shaped this round, priced from its own measured rounds
            // next time.
            if let Some((choice, ms)) = priced {
                // Safety: the stream pointer.
                unsafe {
                    (*s).measured[choice].add(self.clock.elapsed_ms(Mark::Cost) - ms);
                }
            }
        }
        if plan_kind == Kind::Head {
            let ms = self.clock.elapsed_ms(Mark::Overhead);
            crate::trail::event(
                self,
                &[
                    f("ev", Value::Str("overhead")),
                    f("streams", Value::Int(1)),
                    f("rows", Value::Int(rows as i64)),
                    f("ms", crate::events::bits(ms)),
                ],
            )?;
            self.rule.observe_overhead(1, rows as u64, ms)?;
        }
        Ok(Result_ { rows: rows as u32, keep: o.path.len() as u32 })
    }

    /// The stream windows of a shared round in one forward, then each verified,
    /// committed and drafted on its own. Upstream: `roundStreams` (engine.zig:315).
    fn round_streams(&mut self, entries: &[*mut Stream<'a>]) -> Result<(), Error> {
        self.clock.start();
        let arena = self.arena();
        let mut plans: Vec<Plan<'a>> = Vec::with_capacity(entries.len());
        for &s in entries {
            plans.push(self.plan_one(s, None)?);
        }
        windows::allocate(self, &mut plans)?;
        let mut windows_out: Vec<Window<'static>> = Vec::with_capacity(plans.len());
        // Safety: the plan's stream pointers.
        let mut verified: Vec<Verified<'static>> = Vec::with_capacity(plans.len());
        let mut total: u64 = 0;
        for p in &plans {
            let w = windows::build(self, p, false)?;
            let rows = w.rows();
            let sampled: &'static mut [u32] = arena.alloc_slice_copy(&vec![0u32; rows]);
            let drafts: &'static mut [u32] = arena.alloc_slice_copy(&vec![0u32; rows - 1]);
            let rows_lp: &'static mut [LogprobRow] =
                if unsafe { (*p.stream).logprobs.is_some() } {
                    arena.alloc_slice_copy(&vec![LogprobRow::default(); rows])
                } else {
                    &mut []
                };
            total += rows as u64;
            windows_out.push(w);
            verified.push(Verified { sampled, drafts, rows: rows_lp });
        }
        self.backend.verify(&windows_out, &mut verified)?;
        let mut outcomes: Vec<Outcome> = Vec::with_capacity(plans.len());
        let mut paths: Vec<&[u32]> = Vec::with_capacity(plans.len());
        for (p, (w, o)) in plans.iter().zip(windows_out.iter().zip(verified)) {
            // The 'static muts move back out of the Verified, so the round reads
            // what the backend wrote without borrowing the local.
            let Verified { sampled, drafts, rows: rows_lp } = o;
            let tokens = windows::tokens(self, w, &Verified { sampled, drafts, rows: rows_lp });
            let parents = windows::row_parents(self, w)?;
            let sampled: &'static [u32] = sampled;
            let rows_lp: &'static [LogprobRow] = rows_lp;
            let outcome = self.conclude(
                p.stream,
                p.kind,
                p.parents.is_some(),
                p.lanes.is_some(),
                p.position,
                p.forced,
                p.branch_from,
                p.tried,
                p.lanes,
                sampled,
                rows_lp,
                tokens,
                parents,
                w.positions,
            )?;
            paths.push(outcome.path);
            outcomes.push(outcome);
            let rows = w.rows() as u32;
            // Safety: the stream pointer.
            unsafe {
                let x = &mut *p.stream;
                x.min_rows = if x.min_rows == 0 { rows } else { x.min_rows.min(rows) };
            }
        }
        let mut heads: Vec<usize> = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            // Safety: the stream pointer.
            if self.cfg.family_mtp && unsafe { (*p.stream).drafts } {
                heads.push(i);
            }
        }
        let mut whos: Vec<depth::Who> = Vec::with_capacity(heads.len());
        for &i in &heads {
            // Safety: the stream pointer.
            whos.push(unsafe { windows::who(plans[i].stream) });
        }
        let mut budgets = vec![0i64; heads.len()];
        {
            let probe = windows::probe(self);
            self.rule.budgets(&whos, probe, &mut budgets)?;
        }
        let ids: Vec<Value<'_>> = heads
            .iter()
            .map(|&i| Value::Str(unsafe { (*plans[i].stream).id }))
            .collect();
        crate::trail::event(
            self,
            &[
                f("ev", Value::Str("budgets")),
                f("streams", Value::List(arena.alloc_slice_copy(&ids))),
                f("depths", Value::I64s(arena.alloc_slice_copy(&budgets))),
            ],
        )?;
        if !heads.is_empty() && self.cfg.draft_streams {
            // Every stream's head in one batch (the backend drafts a depth for all
            // of them at once).
            let mut requests: Vec<DraftRequest<'_, 'a>> = Vec::with_capacity(heads.len());
            for (i, (&h, &d)) in heads.iter().zip(budgets.iter()).enumerate() {
                let s = plans[h].stream;
                let mut r = DraftRequest::new(
                    s,
                    outcomes[h].follow,
                    plans[h].position,
                    unsafe { (*s).cache_len + 1 },
                    d.max(0) as u32,
                );
                r.rows = Some(outcomes[h].path);
                requests.push(r);
                let _ = i;
            }
            self.ask_drafts(&mut requests)?;
        } else {
            for (&h, &d) in heads.iter().zip(budgets.iter()) {
                self.draft_late(
                    plans[h].stream,
                    plans[h].position,
                    outcomes[h].follow,
                    outcomes[h].path,
                    Some(d),
                )?;
            }
        }
        self.backend.keep(&windows_out, &paths)?;
        let ms = self.clock.elapsed_ms(Mark::Overhead);
        crate::trail::event(
            self,
            &[
                f("ev", Value::Str("overhead")),
                f("streams", Value::Int(plans.len() as i64)),
                f("rows", Value::Int(total as i64)),
                f("ms", crate::events::bits(ms)),
            ],
        )?;
        self.rule.observe_overhead(plans.len() as u32, total, ms)?;
        Ok(())
    }

    /// Commit the target-sampled path (after the thinking budget's cut), each token
    /// with its path row's logprob row. Upstream: `conclude` (engine.zig:368).
    #[allow(clippy::too_many_arguments)]
    fn conclude(
        &mut self,
        s: *mut Stream<'a>,
        kind: Kind,
        parents_present: bool,
        lanes_present: bool,
        _position: u64,
        forced: &'static [u32],
        branch_from: u32,
        tried: bool,
        lanes: Option<*const shape::Shape>,
        sampled: &'static [u32],
        rows_lp: &'static [LogprobRow],
        tokens: &'static [u32],
        parents: &'static [i32],
        _positions: &'static [u64],
    ) -> Result<Outcome, Error> {
        let arena = self.arena();
        // Safety: the stream pointer, held once at a time through the phases.
        let rows = tokens.len();
        let mut path: &'static [u32];
        let mut committed: Vec<u32> = Vec::new();
        if kind == Kind::Forced {
            path = arena.alloc_slice_copy(&(0..rows as u32).collect::<Vec<_>>());
            committed.extend_from_slice(forced);
            // Safety: popForce reads the stream's own queue.
            let tail = unsafe { (*s).pop_force() }.unwrap_or(sampled[rows - 1]);
            committed.push(tail);
        } else {
            path = arena.alloc_slice_copy(&crate::accept::accept_path(tokens, parents, sampled));
            for &r in &path[1..] {
                committed.push(tokens[r as usize]);
            }
            committed.push(sampled[path[path.len() - 1] as usize]);
            if rows > 1 {
                let kept = path.len() - 1;
                self.drafted += rows as u64 - 1;
                self.accepted += kept as u64;
                // Safety: the stream's own counters.
                unsafe {
                    let x = &mut *s;
                    x.drafted += rows as u64 - 1;
                    x.accepted += kept as u64;
                }
                if kind == Kind::Copy {
                    // Safety: the proposer observation.
                    unsafe {
                        if let Some(p) = (*s).proposer.as_mut() {
                            p.observe((rows - 1) as i64, kept as i64);
                        }
                    }
                    if self.alone {
                        // Safety: the copy ramp.
                        unsafe {
                            let x = &mut *s;
                            let width = x.copy_width.unwrap_or(self.cfg.first_copy);
                            x.copy_width = Some(if kept == rows - 1 {
                                self.cfg.max_copy.min(2 * width + 1)
                            } else {
                                self.cfg.first_copy
                            });
                        }
                    }
                } else if kind == Kind::Head {
                    // Safety: the tree observation.
                    unsafe {
                        let x = &mut *s;
                        if let (Some(l), true) = (lanes, x.odds.is_some()) {
                            x.observe_lanes(&*l, path);
                        }
                    }
                    let ds = crate::accept::depths(parents)?;
                    let chain: &[u32] = if branch_from > 0 {
                        &ds[..branch_from as usize]
                    } else {
                        &ds
                    };
                    let mut chain_kept: usize = 0;
                    for &r in &path[1..] {
                        chain_kept += usize::from(branch_from == 0 || r < branch_from);
                    }
                    // Safety: the depth-rule observation.
                    let who = unsafe { windows::who(s) };
                    self.rule.observe_depth(
                        who,
                        chain.iter().copied().max().unwrap_or(0) as i64,
                        chain_kept as i64,
                    )?;
                }
                let mut kept_branch: u64 = 0;
                if branch_from > 0 {
                    // Safety: the branch counters.
                    unsafe {
                        let x = &mut *s;
                        x.branch_rows += rows as u64 - branch_from as u64;
                        for &r in &path[1..] {
                            kept_branch += u64::from(r >= branch_from);
                        }
                        x.branch_accepted += kept_branch;
                    }
                    if tried {
                        let tried_f: f64 = if branch_from > 0 {
                            (rows - branch_from as usize) as f64
                        } else {
                            1.0
                        };
                        // Safety: the graft hit EMA.
                        unsafe {
                            let x = &mut *s;
                            x.graft_hit = 0.75 * x.graft_hit
                                + 0.25 * kept_branch as f64 / tried_f;
                        }
                    }
                }
            }
        }
        // Safety: the thinking budget's cut, over the committed tail.
        let (budget_cut, loop_cut);
        {
            let x = unsafe { &*s };
            budget_cut = x.think_cut(&committed);
            loop_cut = stream::loop_cut(x, &committed);
        }
        let loop_wins = match loop_cut {
            Some(c) => budget_cut.is_none_or(|b| c < b),
            None => false,
        };
        let mut cut: Option<usize> = None;
        if loop_wins {
            let c = loop_cut.unwrap();
            cut = Some(c);
            path = &path[..c + 1];
            committed.truncate(c + 1);
        } else if let Some(c) = budget_cut {
            cut = Some(c);
            path = &path[..c + 1];
            committed.truncate(c);
            // Safety: startClose writes the stream's close queue.
            let close = unsafe { (*s).start_close() };
            committed.push(close);
        }
        // Safety: rounds and the commit itself.
        unsafe {
            (*s).rounds += 1;
        }
        let mut committed_rows: &'static [LogprobRow] = &[];
        if !rows_lp.is_empty() {
            let out: &'static mut [LogprobRow] =
                arena.alloc_slice_copy(&vec![LogprobRow::default(); committed.len()]);
            for (r, (&t, row)) in out
                .iter_mut()
                .zip(committed.iter().zip(path[..committed.len()].iter()))
            {
                *r = rows_lp[*row as usize].for_token(t);
            }
            committed_rows = out;
        }
        // Safety: the commit.
        let landed = unsafe { (*s).commit(&committed, committed_rows)? };
        let got: &'static [u32] = arena.alloc_slice_copy(&committed[..landed]);
        // Safety: the finished trim.
        unsafe {
            if (*s).finished && got.len() < path.len() {
                path = &path[..got.len() + 1];
            }
        }
        let keep = path.len();
        // Safety: cache_len and pending.
        unsafe {
            let x = &mut *s;
            x.cache_len += keep as u64;
            x.pending = Some(committed[keep - 1]);
        }
        let chain = crate::accept::is_chain(parents);
        crate::trail::event(
            self,
            &[
                f("ev", Value::Str("round")),
                f("stream", Value::Str(unsafe { (*s).id })),
                f("kind", Value::Str(kind.name())),
                f("window", Value::U32s(tokens)),
                f(
                    "parents",
                    if chain {
                        Value::Null
                    } else {
                        Value::I32s(parents)
                    },
                ),
                f("positions", Value::U64s(_positions)),
                f("sampled", Value::U32s(sampled)),
                f("forced", Value::U32s(forced)),
                f("path", Value::U32s(path)),
                f("got", Value::U32s(got)),
                f("cut", cut.map(|c| Value::Int(c as i64)).unwrap_or(Value::Null)),
            ],
        )?;
        let _ = (parents_present, lanes_present);
        Ok(Outcome {
            got,
            path,
            cut,
            follow: arena.alloc_slice_copy(&committed[..keep]),
        })
    }

    /// Read the kept rows with their following tokens and ask the head for the next
    /// round's drafts. Upstream: `draftLate` (engine.zig:453).
    fn draft_late(
        &mut self,
        s: *mut Stream<'a>,
        start: u64,
        follow: &'static [u32],
        path: &'static [u32],
        budget: Option<i64>,
    ) -> Result<(), Error> {
        // Safety: the stream pointer.
        let who = unsafe { windows::who(s) };
        let probe = windows::probe(self);
        let d = self.rule.head_depth(&who, budget, probe)? as u32;
        let mut request = DraftRequest::new(s, follow, start, unsafe { (*s).cache_len + 1 }, d);
        request.rows = Some(path);
        self.ask_drafts(&mut [request])
    }

    /// Upstream: `askDrafts` (engine.zig:459).
    fn ask_drafts(&mut self, requests: &mut [DraftRequest<'_, 'a>]) -> Result<(), Error> {
        for r in requests.iter() {
            crate::trail::event(
                self,
                &[
                    f("ev", Value::Str("draft")),
                    f("stream", Value::Str(unsafe { (*r.stream).id })),
                    f("depth", Value::Int(r.depth as i64)),
                    f("position", Value::Int(r.position as i64)),
                    f("follow", Value::U32s(r.follow)),
                    f(
                        "rows",
                        match r.rows {
                            Some(x) => Value::U32s(x),
                            None => Value::Null,
                        },
                    ),
                ],
            )?;
        }
        // A lone stream's lanes: the head tree or chain + grafts with the most
        // expected tokens a round time.
        let mut lone: Option<Box<shape::Shape>> = None;
        let asked_len;
        if self.cfg.head_trees
            && self.cfg.head_lanes > 0
            && self.alone
            && requests.len() == 1
            && requests[0].depth > 0
        {
            asked_len = 1;
            // Safety: the stream's planner state and its round counters, under
            // the crate's invariants.
            unsafe {
                let s = requests[0].stream;
                if (*s).odds.is_none() {
                    (*s).odds = Some(shape::Odds::new(shape::START_ODDS, shape::START_WEIGHT));
                }
                let draft_room = (*s).draft_room();
                let cap = 1.max(
                    shape::MAX_DEPTH
                        .min(self.cfg.head_depth as usize)
                        .min((draft_room - 1).max(1) as usize),
                );
                let odds = (*s).odds.unwrap();
                let costs = Costs {
                    window: &self.cfg.lane_costs,
                    head_ms: self.cfg.mtp_step_ms,
                    over_ms: (*s).round_over.unwrap_or(0.0),
                    tree_over_ms: (*s).tree_over,
                    measured: if self.cfg.own_prices {
                        Some(&(*s).measured)
                    } else {
                        None
                    },
                };
                let grafts = self.graft_prospect(s)?;
                // Every eighth round one size wider than the best, so the odds of
                // deeper lanes stay measured. Upstream's call (engine.zig:472):
                // the head-lane cap is cfg.head_lanes, the depth cap the
                // room-derived cap.
                let p = plan_lanes::pick(
                    &odds,
                    &costs,
                    self.cfg.head_lanes,
                    cap,
                    grafts,
                    (*s).rounds % 8 == 4,
                );
                (*s).picks[p.choice] += 1;
                (*s).graft_room = Some(p.grafts);
                if let Some(ms) = plan_lanes::table_price(&costs, &p) {
                    (*s).priced = Some((p.choice, ms));
                }
                if let Some(t) = p.tree {
                    let t = Box::new(t);
                    // A chain-shaped tree drafts and verifies as the head's chain.
                    let is_chain = t.is_chain();
                    let parents_len = t.parents.len();
                    (*s).held_levels = t.levels();
                    // The request borrows the shape only through the backend's
                    // draft below; the box outlives that call by staying in `lone`
                    // until the stream takes it.
                    let t_ptr: *const shape::Shape = &*t;
                    if is_chain {
                        requests[0].depth = parents_len as u32;
                    } else {
                        requests[0].lanes = Some(&*t_ptr);
                    }
                    lone = Some(t);
                } else {
                    requests[0].depth = p.depth;
                    requests[0].ranks = true;
                    (*s).held_levels = p.depth;
                }
            }
        } else {
            asked_len = requests.len();
            if requests.len() == 1 {
                // Safety: held_levels tracks the ask.
                unsafe {
                    (*requests[0].stream).held_levels = requests[0].depth;
                }
            }
        }
        self.backend.draft(&requests[..asked_len])?;
        let asked: &[DraftRequest<'_, 'a>] = &requests[..asked_len];
        for r in asked {
            // Safety: the stream's held-draft state.
            unsafe {
                let stream = &mut *r.stream;
                stream.drop_held();
                stream.next = Some(Held { count: r.depth, ..Held::default() });
            }
            if let Some(l) = lone.take() {
                // Safety: the stream's lanes.
                unsafe {
                    let stream = &mut *r.stream;
                    if r.lanes.is_some() {
                        stream.next = Some(Held {
                            count: l.parents.len() as u32,
                            tokens: None,
                            parents: Some(l.parents.clone()),
                        });
                    }
                    stream.lanes = Some(l);
                }
            } else if let Some(held) = self.backend.tree(r.stream)? {
                // Safety: the tree head's held drafts.
                unsafe {
                    (*r.stream).next = Some(held);
                }
            }
        }
        Ok(())
    }

    /// The rows a stream's suffix matches would graft this round and the share of
    /// grafted rows it keeps (None: none). Upstream: `graftProspect`
    /// (engine.zig:504).
    fn graft_prospect(&mut self, s: *mut Stream<'a>) -> Result<Option<Grafts>, Error> {
        if self.cfg.fill_lanes == 0 {
            return Ok(None);
        }
        // Safety: the stream's graft state.
        let (rounds, graft_hit) = unsafe { ((*s).rounds, (*s).graft_hit) };
        let probe_round = rounds % 8 == 0;
        if graft_hit < self.cfg.fill_floor && !probe_round {
            return Ok(None);
        }
        let room = self.cfg.fill_lanes.saturating_sub(1) as usize;
        // Safety: the stream's context.
        let found = unsafe { fill::suffix_candidates(&(*s).context, 16, self.cfg.fill_match as usize, room, 8) };
        if found.is_empty() {
            return Ok(None);
        }
        let b = fill::graft(
            fill::Trunk::default(),
            &found,
            room,
        );
        // The stream's recent share of grafted rows kept; every eighth round at
        // least the floor, so grafts get retried.
        let hit = if probe_round {
            graft_hit.max(self.cfg.fill_floor)
        } else {
            graft_hit
        };
        Ok(Some(Grafts { rows: b.tokens.len() as u32, hit }))
    }

    /// One token, the next forward queued first; a copied continuation ahead
    /// switches to verify windows. Upstream: `pipelinedRound` (engine.zig:519).
    fn pipelined_round(&mut self, s: *mut Stream<'a>) -> Result<Result_, Error> {
        // Safety: the stream's pipe state, phase by phase.
        let before = unsafe { (*s).mode };
        let mode = unsafe { (*s).mode }.unwrap_or(Mode::Pipe);
        let mut result = Result_ { rows: 1, keep: 1 };
        if mode == Mode::Verify || mode == Mode::Exit {
            // Safety: a copy proposal ahead.
            let proposal: &'static [u32] = if mode == Mode::Verify && self.cfg.family_width >= 2 {
                windows::copy_proposal(self, s)?
            } else {
                &[]
            };
            if !proposal.is_empty() {
                let got = self.family_round(s, Some(proposal))?;
                // Safety: the mode switch.
                unsafe {
                    let x = &mut *s;
                    if got.keep == 1 || !x.force.is_empty() {
                        x.mode = Some(Mode::Exit);
                    }
                }
            } else {
                // Nothing to copy: back to steps queued ahead (this one lands next
                // round).
                unsafe {
                    let pending = (*s).pending.expect("a piped stream holds a pending token");
                    (*s).mode = Some(Mode::Pipe);
                    self.queue_next(s, Feed::Value(pending))?;
                }
            }
            crate::trail::pipe(self, s, before, &[])?;
            return Ok(result);
        }
        let mut current = Feed::Handle(unsafe { (*s).inflight }.expect("a piped stream holds a queued draw"));
        unsafe { (*s).inflight = None };
        if let Some(t) = self.forced_next(s)? {
            current = Feed::Value(t);
        }
        let token: u32;
        if mode == Mode::Drain {
            // The last queued step: no new one.
            token = self.read_feed(current)?;
            unsafe { (*s).mode = Some(Mode::Verify) };
        } else {
            // The GPU starts the next step first.
            self.queue_next(s, current)?;
            token = self.read_feed(current)?;
        }
        // Safety: the commit.
        unsafe {
            (*s).rounds += 1;
        }
        let landed = unsafe { (*s).commit(&[token], &[])? };
        unsafe { (*s).pending = Some(token) };
        unsafe {
            let x = &mut *s;
            if self.cfg.family_width >= 2
                && x.mode.unwrap_or(Mode::Pipe) == Mode::Pipe
                && !x.finished
                && x.drafts
            {
                let ahead = windows::copy_proposal(self, s)?;
                if !ahead.is_empty() {
                    // A copy window is ahead: land the queued step.
                    x.mode = Some(Mode::Drain);
                }
            }
        }
        // Safety: the landed tail.
        let got: &'static [u32] = unsafe {
            let emitted = (*s).emitted();
            self.arena()
                .alloc_slice_copy(&emitted[emitted.len() - landed..])
        };
        crate::trail::pipe(self, s, before, got)?;
        result.keep = 1;
        Ok(result)
    }

    /// Land the queued token without queuing another (a shared round is
    /// synchronous). Upstream: `landInflight` (engine.zig:558).
    fn land_inflight(&mut self, s: *mut Stream<'a>) -> Result<(), Error> {
        let mut current = Feed::Handle(unsafe { (*s).inflight }.expect("the stream holds a queued draw"));
        unsafe { (*s).inflight = None };
        if let Some(t) = self.forced_next(s)? {
            current = Feed::Value(t);
        }
        let token = self.read_feed(current)?;
        unsafe {
            (*s).rounds += 1;
        }
        let landed = unsafe { (*s).commit(&[token], &[])? };
        unsafe { (*s).pending = Some(token) };
        unsafe { (*s).mode = Some(Mode::Verify) };
        // Safety: the landed tail, for the trail.
        let got: &'static [u32] = unsafe {
            let emitted = (*s).emitted();
            self.arena()
                .alloc_slice_copy(&emitted[emitted.len() - landed..])
        };
        crate::trail::event(
            self,
            &[
                f("ev", Value::Str("land")),
                f("stream", Value::Str(unsafe { (*s).id })),
                f("got", Value::U32s(got)),
            ],
        )?;
        Ok(())
    }

    /// Feed a token and queue the draw of the next at the new cache length.
    /// Upstream: `queueNext` (engine.zig:571).
    fn queue_next(&mut self, s: *mut Stream<'a>, feed: Feed) -> Result<(), Error> {
        unsafe {
            (*s).cache_len += 1;
            let position = (*s).cache_len;
            let h = self.backend.queue(s, feed, position)?;
            (*s).inflight = Some(h);
            if let Some(log) = &mut self.log {
                let i = log.add(&[
                    f("ev", Value::Str("queue")),
                    f("stream", Value::Str((*s).id)),
                    f("position", Value::Int(position as i64)),
                    f("token", Value::Null),
                ]);
                self.unread.push(Unread { event: i, handle: h });
            }
        }
        Ok(())
    }

    /// The thinking budget's or a forced fix's token at the next position instead
    /// of the draw. Upstream: `forcedNext` (engine.zig:582).
    fn forced_next(&mut self, s: *mut Stream<'a>) -> Result<Option<u32>, Error> {
        // Safety: the stream's own queues.
        unsafe {
            if let Some(t) = (*s).pop_force() {
                return Ok(Some(t));
            }
            if (*s).cuts_next() {
                return Ok(Some((*s).start_close()));
            }
        }
        Ok(None)
    }

    /// Upstream: `readFeed`.
    fn read_feed(&mut self, feed: Feed) -> Result<u32, Error> {
        match feed {
            Feed::Handle(h) => self.backend.read(h),
            Feed::Value(v) => Ok(v),
        }
    }

    fn release(&mut self, s: *mut Stream<'a>) {
        // Safety: the stream's round state.
        unsafe {
            (*s).drop_round_state();
        }
        self.backend.release(s);
    }

    /// `windows::plan` without the middle-man borrow: the engine method callers
    /// upstream reach through `win.plan(e, s, copied)`.
    fn plan_one(
        &mut self,
        s: *mut Stream<'a>,
        copied: Option<&'static [u32]>,
    ) -> Result<Plan<'a>, Error> {
        windows::plan(self, s, copied)
    }
}
