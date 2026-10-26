//! A round's windows (Python `_plan_window`, `_copy_proposal`, `_allocate`):
//! forced, copied or drafted rows.
//!
//! Port of `zig/src/core/lanes/windows.zig`.

use crate::accept;
use crate::allocate;
use crate::backend::{Alternative, Verified, Window};
use crate::depth::{Probe, Who};
use crate::engine::Engine;
use crate::events::{f, Value};
use crate::fill;
use crate::shape;
use crate::stream::Stream;
use crate::Error;

/// Upstream: `windows.zig:16  pub const Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Forced,
    Copy,
    Head,
    Fill,
    None,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Forced => "forced",
            Kind::Copy => "copy",
            Kind::Head => "head",
            Kind::Fill => "fill",
            Kind::None => "none",
        }
    }
}

/// One stream's window as planned, then as allocated (Python's plan lists).
/// Upstream: `windows.zig:19  pub const Plan`.
pub struct Plan<'s> {
    pub stream: *mut Stream<'s>,
    /// The stream's cache length when the round started.
    pub position: u64,
    pub kind: Kind,
    /// Head drafts the backend holds.
    pub held: u32,
    /// Drafts known on the host (forced, copied, tree).
    pub tokens: &'static [u32],
    pub forced: &'static [u32],
    /// A tree's drafts' parents.
    pub parents: Option<&'static [i32]>,
    /// The first row of suffix-match branches (0: none; fill.zig).
    pub branch_from: u32,
    /// The planner read the head's chain back to graft branches.
    pub tried: bool,
    /// The held drafts' tree shape (each lane's depth and rank).
    pub lanes: Option<*const shape::Shape>,
}

impl Plan<'_> {
    /// Upstream: `count`.
    pub fn count(&self) -> usize {
        self.held as usize + self.tokens.len()
    }
}

/// Forced tokens first, then a copied continuation, then the head's drafts
/// (Python `_plan_window`). Upstream: `windows.zig:37  pub fn plan`.
///
/// Safety: `s` per the crate invariants; the returned slices live in the engine's
/// arena (Engine::arena's contract).
pub fn plan<'e, 's>(
    e: &mut Engine<'e>,
    s: *mut Stream<'s>,
    copied: Option<&[u32]>,
) -> Result<Plan<'s>, Error>
where
    's: 'e,
{
    let arena = e.arena();
    // Safety: the stream pointer, under the crate invariants.
    let stream = unsafe { &mut *s };
    let mut p = Plan {
        stream: s,
        position: stream.cache_len,
        kind: Kind::None,
        held: 0,
        tokens: &[],
        forced: &[],
        parents: None,
        branch_from: 0,
        tried: false,
        lanes: None,
    };
    let queued = stream.next.take();
    if !stream.force.is_empty() {
        let width: usize = if stream.drafts {
            e.cfg.base_width.min(e.cfg.batch_rows) as usize
        } else {
            1
        };
        let take = (width - 1).min(stream.force.len());
        let forced: &'static [u32] = arena.alloc_slice_copy(&stream.force[..take]);
        stream.force.drain(..take);
        p.kind = Kind::Forced;
        p.forced = forced;
        p.tokens = forced;
    } else {
        let copy: &[u32] = match copied {
            Some(c) => c,
            None => {
                if stream.drafts {
                    copy_proposal(e, s)?
                } else {
                    &[]
                }
            }
        };
        if !copy.is_empty() {
            p.kind = Kind::Copy;
            p.tokens = arena.alloc_slice_copy(copy);
        } else if stream.drafts && queued.is_some() {
            let q = queued.unwrap();
            if let Some(tokens) = &q.tokens {
                p.tokens = arena.alloc_slice_copy(tokens);
                if let Some(parents) = &q.parents {
                    if !accept::is_chain(parents) {
                        p.parents = Some(arena.alloc_slice_copy(parents));
                    }
                }
            } else {
                p.held = q.count;
                if let Some(l) = &stream.lanes {
                    // A held tree is a lone stream's (shared rounds verify chains);
                    // a chain's lanes are observed too.
                    if let Some(parents) = &q.parents {
                        if e.alone {
                            p.parents = Some(arena.alloc_slice_copy(parents));
                        } else {
                            p.held = 0;
                        }
                    }
                    if p.held > 0 {
                        p.lanes = Some(&**l as *const shape::Shape);
                    }
                }
            }
            if p.count() > 0 {
                p.kind = Kind::Head;
            }
        }
    }
    if e.cfg.fill_lanes > 0
        && stream.drafts
        && p.kind != Kind::Forced
        && (p.parents.is_none() || p.lanes.is_some())
    {
        fill_lanes(e, s, &mut p)?;
    }
    crate::trail::event(
        e,
        &[
            f("ev", Value::Str("plan")),
            f("stream", Value::Str(stream.id)),
            f("kind", Value::Str(p.kind.name())),
            f("n", Value::Int(p.count() as i64)),
            f("tree", Value::Bool(p.parents.is_some())),
        ],
    )?;
    Ok(p)
}

/// Suffix-match branches grafted onto the planned drafts, up to cfg.fill_lanes
/// rows. Upstream: `windows.zig:86  fn fillLanes`.
fn fill_lanes<'e, 's>(
    e: &mut Engine<'e>,
    s: *mut Stream<'s>,
    p: &mut Plan<'s>,
) -> Result<(), Error> {
    let arena = e.arena();
    // Safety: the stream pointer, per the crate invariants.
    let stream = unsafe { &mut *s };
    let first = p.count();
    let room = (e.cfg.fill_lanes as usize)
        .saturating_sub(1 + first)
        .min(stream.graft_room.map(|g| g as usize).unwrap_or(usize::MAX));
    if room == 0 {
        return Ok(());
    }
    let found = fill::suffix_candidates(&stream.context, 16, e.cfg.fill_match as usize, room, 8);
    if found.is_empty() || (stream.graft_hit < e.cfg.fill_floor && stream.rounds % 8 != 0) {
        return Ok(());
    }
    // The trunk's lanes: the drafts in order for a chain, a tree's rank-0 path from
    // the pending row.
    let trunk_lanes: Vec<usize>;
    let levels = match p.lanes {
        Some(l) => {
            let mut lanes = vec![0usize; shape::MAX_DEPTH * shape::RANKS * 4];
            // Safety: the shape is the stream's own lanes, alive under the crate's
            // invariants.
            let n = shape::chain_lanes(unsafe { &*l }, &mut lanes);
            trunk_lanes = lanes;
            n
        }
        None => {
            trunk_lanes = (0..first).collect();
            first
        }
    };
    let Some(trunk) = head_trunk(e, s, p, levels)? else {
        return Ok(());
    };
    p.tried = p.held > 0;
    let b = fill::graft(
        fill::Trunk {
            tokens: &trunk.0,
            others: &trunk.1,
        },
        &found,
        room,
    );
    if b.tokens.is_empty() {
        return Ok(());
    }
    let mut parents: Vec<i32> = Vec::with_capacity(first + b.parents.len());
    if let Some(held) = p.parents {
        parents.extend_from_slice(held);
    } else {
        for k in 0..first {
            parents.push(k as i32 - 1);
        }
    }
    // A graft's parent: the pending row, a trunk level (that level's lane), or one
    // of its own rows (after the drafts).
    for &g in &b.parents {
        parents.push(if g < 0 {
            -1
        } else if (g as usize) < levels {
            trunk_lanes[g as usize] as i32
        } else {
            (first + g as usize - levels) as i32
        });
    }
    if p.held > 0 {
        p.tokens = arena.alloc_slice_copy(&b.tokens);
    } else {
        let joined: Vec<u32> = p.tokens.iter().copied().chain(b.tokens.iter().copied()).collect();
        p.tokens = arena.alloc_slice_copy(&joined);
    }
    p.parents = Some(arena.alloc_slice_copy(&parents));
    p.branch_from = (1 + first) as u32;
    if p.kind == Kind::None {
        p.kind = Kind::Fill;
    }
    Ok(())
}

/// The planned drafts' trunk of `levels` lanes: host drafts as they are, held ones
/// with the head's ranks (None: unread). Upstream: `windows.zig:118  headTrunk`.
#[allow(clippy::type_complexity)]
fn head_trunk(
    e: &mut Engine<'_>,
    s: *mut Stream<'_>,
    p: &Plan<'_>,
    levels: usize,
) -> Result<Option<(Vec<u32>, Vec<[u32; 3]>)>, Error> {
    if p.held == 0 {
        return Ok(Some((p.tokens.to_vec(), Vec::new())));
    }
    let mut alts: Vec<Alternative> = (0..levels)
        .map(|_| Alternative { tokens: [0; 4], probs: [0.0; 4] })
        .collect();
    let read = e.backend.alternatives(s, &mut alts[..])?;
    if read < levels {
        return Ok(None);
    }
    let mut chain: Vec<u32> = Vec::with_capacity(levels);
    let mut others: Vec<[u32; 3]> = Vec::with_capacity(levels);
    for x in &alts[..levels] {
        chain.push(x.tokens[0]);
        others.push([x.tokens[1], x.tokens[2], x.tokens[3]]);
    }
    Ok(Some((chain, others)))
}

/// Copied spans backed by `enter_match` matching tokens (Python
/// `_copy_proposal`); the proposer owns them. Upstream: `windows.zig:134
/// pub fn copyProposal`.
///
/// Safety: `s` per the crate invariants. The returned slice is the proposer's own
/// buffer, valid until its next `propose` — upstream's exact contract — so the
/// unbounded lifetime matches it.
pub fn copy_proposal(e: &mut Engine<'_>, s: *mut Stream<'_>) -> Result<&'static [u32], Error> {
    // The proposer's continuation is "valid until its next propose" (upstream's
    // own contract, proposer.zig:17); the unbounded return mirrors it. Sound
    // because the round consumes the slice before any further engine call.
    // Safety: the stream and its proposer, under the crate invariants. The
    // proposer's raw pointer keeps its &mut borrow apart from the reads of the
    // stream's other fields, the way upstream's ctx pointer does; the proposer
    // does not re-enter the stream.
    unsafe {
        let stream = &mut *s;
        let Some(proposer) = stream.proposer.as_deref_mut() else {
            return Ok(&[]);
        };
        if e.cfg.max_copy == 0 || !stream.force.is_empty() {
            return Ok(&[]);
        }
        let width: u32 = if e.alone {
            stream.copy_width.unwrap_or(e.cfg.first_copy)
        } else {
            e.cfg.first_copy
        };
        // draftRoom, by field: max_new less what is emitted so far.
        let draft_room = stream.max_new as i64 - (stream.context.len() - stream.prompt_len) as i64;
        let max_draft = (width as i64).min(draft_room - 1);
        let proposer: *mut dyn crate::proposer::Proposer = proposer;
        // The proposer's own contract keeps its buffer valid until the next
        // propose (proposer.zig:17); the transmute mirrors that lifetime the way
        // upstream's pointer parameters do.
        let copied: &'static [u32] =
            std::mem::transmute::<&[u32], &'static [u32]>(match (*proposer)
                .propose(&stream.context, max_draft)
            {
                Ok(c) => c,
                Err(_) => return Ok(&[]),
            });
        let len = copied.len();
        let last_match = (*proposer).last_match();
        if len < 2 || last_match < e.cfg.enter_match {
            return Ok(&[]);
        }
        Ok(copied)
    }
}

/// Upstream: `windows.zig:143  pub fn copyAhead` — the depth rule's probe body.
///
/// # Safety
///
/// `ptr` is the engine `probe` built; `who.stream` its own stream.
pub unsafe fn copy_ahead(ptr: *mut core::ffi::c_void, who: *mut Who) -> bool {
    // Safety: the probe pairs this engine with its streams by construction.
    unsafe {
        let e = &mut *(ptr as *mut Engine<'_>);
        let stream = (*who).stream.expect("the probe sees a stream");
        copy_proposal(e, stream).map(|c| !c.is_empty()).unwrap_or(false)
    }
}

/// Upstream: `windows.zig:150  pub fn probe`.
pub fn probe(e: &mut Engine<'_>) -> Probe {
    Probe {
        ptr: e as *mut Engine<'_> as *mut core::ffi::c_void,
        copy: copy_ahead,
    }
}

/// Upstream: `windows.zig:154  pub fn who`.
///
/// # Safety
///
/// `s` per the crate invariants; the Who points into the stream's own state.
pub unsafe fn who(s: *mut Stream<'_>) -> Who {
    unsafe {
        let stream = &mut *s;
        // The stream pointer erases to 'static like every cross-link in the core
        // (the crate's invariants own its lifetime; see lib.rs).
        let s: *mut Stream<'static> = std::mem::transmute(s);
        Who {
            state: &mut stream.depth,
            draft_room: stream.draft_room(),
            finished: stream.finished,
            forced: !stream.force.is_empty(),
            granted: stream.granted,
            stream: Some(s),
        }
    }
}

/// Trim draft prefixes by landing probability to the shared width with the most
/// expected tokens a ms. Upstream: `windows.zig:159  pub fn allocate`.
pub fn allocate(e: &mut Engine<'_>, plans: &mut [Plan<'_>]) -> Result<(), Error> {
    let arena = e.arena();
    let mut fixed: Vec<u32> = Vec::with_capacity(plans.len());
    let mut owned: Vec<Vec<f64>> = Vec::with_capacity(plans.len());
    for p in plans.iter() {
        let n = p.count();
        if p.kind == Kind::Forced || n == 0 {
            fixed.push(1 + if p.kind == Kind::Forced { n as u32 } else { 0 });
            owned.push(Vec::new());
        } else if p.kind == Kind::Copy {
            fixed.push(1);
            owned.push(allocate::chain_probabilities(&[e.cfg.copy_rate], n));
        } else {
            fixed.push(1);
            owned.push(Vec::new());
        }
    }
    // Head plans' node chances need the engine mutably, so they come after the
    // fixed pass (the copy chances are already in `owned`).
    for (i, p) in plans.iter().enumerate() {
        if p.kind != Kind::Forced && p.kind != Kind::Copy && p.count() > 0 {
            owned[i] = node_chances(e, p.stream, p.count())?;
        }
    }
    let probs: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
    let overhead = e.rule.overhead_ms(plans.len());
    let counts = allocate::allocate(
        &fixed,
        &probs,
        &e.cfg.shared_costs,
        overhead,
        plans.len().max(e.cfg.batch_rows as usize) as u64,
    );
    for (p, (&chances, &c)) in plans.iter_mut().zip(probs.iter().zip(counts.iter())) {
        if p.kind == Kind::Head {
            // Safety: the stream pointer.
            unsafe {
                (*p.stream).granted = Some(c);
            }
        }
        if chances.is_empty() || c == chances.len() as u32 {
            continue;
        }
        if c == 0 {
            p.kind = Kind::None;
            p.held = 0;
            p.tokens = &[];
            p.parents = None;
        } else if let Some(parents) = p.parents {
            let tree = accept::sanitize_tree(p.tokens, parents, c as usize);
            p.tokens = arena.alloc_slice_copy(&tree.tokens);
            p.parents = Some(arena.alloc_slice_copy(&tree.parents));
        } else if p.held > 0 {
            p.held = c;
        } else {
            p.tokens = &p.tokens[..c as usize];
        }
    }
    let rows: Vec<Value<'_>> = plans
        .iter()
        .map(|p| {
            Value::List(arena.alloc_slice_copy(&[
                Value::Str(p.kind.name()),
                Value::Int(p.count() as i64),
            ]))
        })
        .collect();
    crate::trail::event(
        e,
        &[
            f("ev", Value::Str("alloc")),
            f("plans", Value::List(arena.alloc_slice_copy(&rows))),
        ],
    )?;
    Ok(())
}

/// The head's chance for each held draft, else chain chances from the stream's
/// per-depth acceptance. Upstream: `windows.zig:204  pub fn nodeChances`.
fn node_chances(e: &mut Engine<'_>, s: *mut Stream<'_>, n: usize) -> Result<Vec<f64>, Error> {
    if e.cfg.node_probabilities {
        let mut out = vec![0.0f64; n];
        if e.backend.probabilities(s, &mut out)? {
            return Ok(out);
        }
    }
    // Safety: the stream pointer.
    let who = unsafe { who(s) };
    let rates = e.rule.rates(who)?.to_vec();
    Ok(allocate::chain_probabilities(&rates, n))
}

/// The window's rows, parents and keyed positions, ready for the backend.
/// Upstream: `windows.zig:216  pub fn build`.
///
/// Safety: the plan's stream pointer; the window's slices live in the arena.
pub fn build<'e, 's>(e: &mut Engine<'e>, p: &Plan<'s>, early: bool) -> Result<Window<'static>, Error> {
    let arena = e.arena();
    let rows = 1 + p.count();
    let parent_rows = accept::row_parents(rows, p.parents);
    let ds = accept::depths(&parent_rows)?;
    let parents: &'static [i32] = if p.parents.is_some() {
        arena.alloc_slice_copy(&parent_rows)
    } else {
        &[]
    };
    let positions: &'static mut [u64] = arena.alloc_slice_copy(&vec![0u64; rows]);
    for (x, d) in positions.iter_mut().zip(&ds) {
        *x = p.position + 1 + *d as u64;
    }
    // Safety: the stream pointer.
    let pending = unsafe { (*p.stream).pending.expect("a round's stream holds a pending token") };
    // The window's stream pointer erases like every cross-link in the core.
    let stream: *mut Stream<'static> = unsafe { std::mem::transmute(p.stream) };
    Ok(Window {
        stream,
        pending,
        held: p.held,
        tokens: p.tokens,
        parents: if p.parents.is_some() { Some(parents) } else { None },
        positions,
        early,
    })
}

/// Upstream: `windows.zig:226  pub fn rowParents`.
pub fn row_parents(e: &mut Engine<'_>, w: &Window<'_>) -> Result<&'static [i32], Error> {
    match w.parents {
        // The window's arena slices are unbounded (Engine::arena's contract).
        Some(p) => Ok(unsafe { std::mem::transmute::<&[i32], &'static [i32]>(p) }),
        None => Ok(e.arena().alloc_slice_copy(&accept::row_parents(w.rows(), None))),
    }
}

/// The rows' tokens: the pending one, then the drafts (held drafts as the backend
/// read them back). Upstream: `windows.zig:232  pub fn tokens`.
pub fn tokens(e: &mut Engine<'_>, w: &Window<'_>, o: &Verified<'_>) -> &'static [u32] {
    let arena = e.arena();
    let out: &'static mut [u32] = arena.alloc_slice_copy(&vec![0u32; w.rows()]);
    out[0] = w.pending;
    out[1..1 + w.held as usize].copy_from_slice(&o.drafts[..w.held as usize]);
    out[1 + w.held as usize..].copy_from_slice(w.tokens);
    out
}
