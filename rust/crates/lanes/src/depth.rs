//! Draft depth by expected tokens a millisecond (Python family_depth.py DraftDepth
//! and FamilyRounds._head_depth).
//!
//! Port of `zig/src/core/lanes/depth.zig`.

use crate::allocate;
use crate::stream::Stream;

use crate::table::Table;
use crate::Error;

/// Upstream: `depth.zig:7  pub const Constants`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constants {
    pub depth_rate: f64, // weight of the newest round in a depth's acceptance
    pub depth_probe_every: u64, // rounds between probes one depth deeper
    pub cost_rate: f64, // weight of the newest round in a depth's measured time
    pub plain_margin: f64, // a drafted depth must beat plain rounds' tokens a ms by this much
    pub plain_wait_most: u64, // plain rounds between probes at most
    pub draft_slack: i64, // drafts a stream offers past what its last shared round granted it
}

impl Default for Constants {
    fn default() -> Self {
        Constants {
            depth_rate: 0.15,
            depth_probe_every: 8,
            cost_rate: 0.2,
            plain_margin: 1.05,
            plain_wait_most: 128,
            draft_slack: 2,
        }
    }
}

/// A stream's acceptance by depth and probe counters (Python
/// `_depth_state[stream]`). Upstream: `depth.zig:17  pub const State`.
pub struct State {
    pub p: Vec<f64>,
    pub rounds: u64,
    pub plain: Option<u64>,
    pub wait: Option<u64>,
    /// plain_guard: the stream's own round times.
    pub ms: Option<Table>,
}

impl State {
    fn new(prior: &[f64], most: usize) -> State {
        State {
            p: prior[..prior.len().min(most.max(1))].to_vec(),
            rounds: 0,
            plain: None,
            wait: None,
            ms: None,
        }
    }
}

/// What the rule reads of a stream when it decides. Upstream: `depth.zig:31
/// pub const Who` — the state is the stream's own `depth` field, reached through a
/// raw pointer exactly as upstream reaches it through `*?State`.
///
/// Safety: `state` must point at a `Stream`'s `depth` field, `stream` at that
/// stream, both alive and untouched elsewhere while the rule runs (the engine's
/// single-thread invariant, lib.rs).
#[derive(Clone, Copy)]
pub struct Who {
    pub state: *mut Option<State>,
    pub draft_room: i64,
    pub finished: bool,
    pub forced: bool,
    pub granted: Option<u32>,
    pub stream: Option<*mut Stream<'static>>,
}

/// Whether a copied continuation is ahead of a stream (the round loop proposes;
/// Python `_copy_proposal`). Upstream: `depth.zig:41  Probe` — the same
/// pointer-callback pair, under the crate's single-thread invariants.
#[derive(Clone, Copy)]
pub struct Probe {
    pub ptr: *mut core::ffi::c_void,
    pub copy: unsafe fn(*mut core::ffi::c_void, *mut Who) -> bool,
}

/// Upstream: `depth.zig:46  pub const Overhead`.
#[derive(Clone, Copy)]
pub struct Overhead {
    pub streams: u32,
    pub ms: f64,
}

/// Upstream: `depth.zig:48  pub const Rule`.
pub struct Rule {
    pub k: Constants,
    pub prior: Vec<f64>,
    pub most_drafts: u32,
    pub family_costs: *const Table,
    pub shared_costs: *const Table,
    pub mtp_step_ms: f64,
    pub plain_guard: bool,
    pub node_probabilities: bool,
    pub batch_rows: u32,
    /// A round's wall time by drafts, shared by every stream.
    pub round_ms: Table,
    /// A shared round's ms past its forward, by streams (insertion order).
    pub overhead: Vec<Overhead>,
}

impl Rule {
    /// Upstream: `Engine.init` builds the rule from the config's fields
    /// (engine.zig:50-61).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: Constants,
        prior: &[f64],
        most_drafts: u32,
        family_costs: *const Table,
        shared_costs: *const Table,
        mtp_step_ms: f64,
        plain_guard: bool,
        node_probabilities: bool,
        batch_rows: u32,
    ) -> Rule {
        Rule {
            k,
            prior: prior.to_vec(),
            most_drafts,
            family_costs,
            shared_costs,
            mtp_step_ms,
            plain_guard,
            node_probabilities,
            batch_rows,
            round_ms: Table::new(),
            overhead: Vec::new(),
        }
    }

    /// Upstream: `rates` — the stream's own acceptance, seeded from the prior.
    ///
    /// Safety: `who.state` points into a live stream (crate invariants).
    pub fn rates(&self, who: Who) -> Result<&[f64], Error> {
        unsafe {
            let slot = &mut *who.state;
            if slot.is_none() {
                let n = self.prior.len().min((self.most_drafts as usize).max(1));
                *slot = Some(State::new(&self.prior, n));
            }
            Ok(&slot.as_ref().unwrap().p)
        }
    }

    /// Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves
    /// toward whether it was kept. Upstream: `observeDepth`.
    pub fn observe_depth(&mut self, who: Who, proposed: i64, accepted: i64) -> Result<(), Error> {
        // Safety: the rule mutates the stream's own depth state under the engine's
        // single-thread invariant.
        let p_len;
        {
            let p = self.rates(who)?;
            p_len = p.len();
        }
        unsafe {
            let p = (*who.state).as_mut().unwrap().p.as_mut_slice();
            let n = proposed.clamp(0, p_len as i64);
            let mut j: i64 = 0;
            while j < n {
                if accepted < j {
                    break;
                }
                let hit: f64 = if accepted > j { 1.0 } else { 0.0 };
                let at = j as usize;
                p[at] += self.k.depth_rate * (hit - p[at]);
                j += 1;
            }
        }
        Ok(())
    }

    /// Round times by depth: the stream's own under `plain_guard` (its sampling
    /// sets a row's host work), else shared. Upstream: `costs`.
    ///
    /// Safety: `who.state` per crate invariants.
    fn costs(&self, who: Option<Who>) -> Result<*mut Table, Error> {
        match who {
            None => Ok(&self.round_ms as *const Table as *mut Table),
            Some(w) => {
                if !self.plain_guard {
                    return Ok(&self.round_ms as *const Table as *mut Table);
                }
                self.rates(w)?;
                unsafe {
                    let slot = &mut *w.state;
                    let st = slot.as_mut().unwrap();
                    if st.ms.is_none() {
                        st.ms = Some(Table::new());
                    }
                    Ok(st.ms.as_mut().unwrap() as *mut Table)
                }
            }
        }
    }

    /// A stream's first round carries the prefill-to-decode switch: its time stays
    /// out of the costs. Upstream: `observeCost`.
    pub fn observe_cost(
        &mut self,
        drafts: i64,
        ms: f64,
        initializing: bool,
        who: Option<Who>,
    ) -> Result<(), Error> {
        if drafts < 0 || (drafts == 0 && !self.plain_guard) || initializing {
            return Ok(());
        }
        // Safety: the table pointer is either the rule's own or the stream's own
        // depth state, both alive under the crate invariants.
        unsafe {
            let table = self.costs(who)?;
            let before = (*table).get(drafts);
            (*table).put(drafts, before.map_or(ms, |b| b + self.k.cost_rate * (ms - b)));
        }
        Ok(())
    }

    /// Measured round time when there is one, else the load-time forward cost plus
    /// head steps. Upstream: `roundCost`.
    pub fn round_cost(&mut self, drafts: i64, who: Option<Who>) -> Result<Option<f64>, Error> {
        // Safety: as `costs`.
        let table = self.costs(who)?;
        unsafe {
            if let Some(v) = (*table).get(drafts) {
                return Ok(Some(v));
            }
            let Some(forward) = (*self.family_costs).get(drafts + 1) else {
                return Ok(None);
            };
            let mut modeled = forward + self.mtp_step_ms * drafts as f64;
            if self.plain_guard {
                // Untimed: its model plus the least time a row of a timed round
                // took beyond its model.
                let mut least: Option<f64> = None;
                for (d, slot) in (*table).slots().iter().enumerate() {
                    let Some(ms) = *slot else { continue };
                    let Some(fc) = (*self.family_costs).get(d as i64 + 1) else {
                        continue;
                    };
                    let beyond = (ms - fc - self.mtp_step_ms * d as f64) / (d as f64 + 1.0);
                    least = Some(match least {
                        Some(l) => beyond.min(l),
                        None => beyond,
                    });
                }
                let floor = least.unwrap_or(0.0);
                modeled += (drafts as f64 + 1.0) * floor.max(0.0);
            }
            Ok(Some(modeled))
        }
    }

    /// The most expected tokens a ms, probing one depth farther every
    /// `depth_probe_every` rounds. Upstream: `depth`.
    pub fn depth(&mut self, who: Who) -> Result<i64, Error> {
        let most = (self.most_drafts as i64).min((who.draft_room - 1).max(1));
        if most <= 0 {
            return Ok(0);
        }
        let p_len;
        let p0;
        {
            let p = self.rates(who)?;
            p_len = p.len();
            p0 = p[0];
        }
        // Safety: family_costs is the config's table, alive as long as the engine.
        let family_empty = unsafe { (*self.family_costs).is_empty() };
        if family_empty {
            let want: i64 = if p0 < 0.8 { 1 } else if p0 < 0.9 { 2 } else { 3 };
            return Ok(1.max(most.min(want)));
        }
        let mut best: i64 = 1;
        let mut best_rate: f64 = -1.0;
        if self.plain_guard {
            if let Some(plain) = self.round_cost(0, Some(who))? {
                if plain != 0.0 {
                    // A draft must beat plain by plain_margin: estimates near the
                    // line are noise.
                    best = 0;
                    best_rate = self.k.plain_margin / plain;
                }
            }
        }
        let mut expected: f64 = 1.0;
        let mut run: f64 = 1.0;
        let mut d: i64 = 1;
        while d <= most {
            let Some(cost) = self.round_cost(d, Some(who))? else { break };
            let at = (d - 1) as usize;
            // Safety: the stream's own acceptance, under the crate invariants.
            unsafe {
                let p = (*who.state).as_ref().unwrap().p.as_slice();
                run *= if at < p_len { p[at] } else { p[p_len - 1] };
            }
            expected += run;
            if expected / cost > best_rate {
                best = d;
                best_rate = expected / cost;
            }
            d += 1;
        }
        // Safety: the probe counters live in the stream's depth state.
        unsafe {
            let st = (*who.state).as_mut().unwrap();
            st.rounds += 1;
            if best == 0 {
                return Ok(i64::from(self.probe_plain(st)));
            }
            st.plain = Some(0);
            st.wait = Some(self.k.depth_probe_every);
            if best < most && st.rounds % self.k.depth_probe_every == 0 {
                best += 1;
            }
        }
        Ok(best)
    }

    /// One draft after `wait` plain rounds in a row; each probe doubles the wait
    /// until drafting wins again. Upstream: `probePlain`.
    fn probe_plain(&self, st: &mut State) -> bool {
        st.plain = Some(st.plain.unwrap_or(0) + 1);
        let wait = st.wait.unwrap_or(self.k.depth_probe_every);
        if st.plain.unwrap() < wait {
            return false;
        }
        st.plain = Some(0);
        st.wait = Some((2 * wait).min(self.k.plain_wait_most));
        true
    }

    /// Python `_head_depth`: the next round's head drafts for a stream (0 when
    /// forced, 1 with a copy ahead). Upstream: `headDepth`.
    pub fn head_depth(&mut self, who: &Who, budget: Option<i64>, probe: Probe) -> Result<i64, Error> {
        let d: i64 = if who.finished {
            0
        } else if let Some(b) = budget {
            b
        } else if self.node_probabilities {
            (self.most_drafts as i64).min((who.draft_room - 1).max(1))
        } else {
            self.depth(*who)?
        };
        if who.forced {
            return Ok(0);
        }
        if d != 0 {
            // Safety: the probe is the engine's own copy-ahead check, running
            // under the crate's single-thread invariants.
            unsafe {
                if (probe.copy)(probe.ptr, who as *const Who as *mut Who) {
                    return Ok(1);
                }
            }
        }
        Ok(d)
    }

    /// Head drafts for the streams of a shared round, sized by allocation over
    /// their chain chances. Upstream: `budgets`.
    pub fn budgets(&mut self, whos: &[Who], probe: Probe, out: &mut [i64]) -> Result<(), Error> {
        if self.node_probabilities {
            for (w, o) in whos.iter().zip(out.iter_mut()) {
                let inner = self.head_depth(w, None, probe)?;
                let granted = w.granted.map(|g| g as i64).unwrap_or(self.most_drafts as i64);
                *o = self.head_depth(w, Some(inner.min(granted + self.k.draft_slack)), probe)?;
            }
            return Ok(());
        }
        let mut most = vec![0i64; whos.len()];
        let mut probs: Vec<Vec<f64>> = Vec::with_capacity(whos.len());
        for (i, w) in whos.iter().enumerate() {
            most[i] = (self.most_drafts as i64).min((w.draft_room - 1).max(1));
            let rates = self.rates(*w)?.to_vec();
            probs.push(allocate::chain_probabilities(&rates, most[i] as usize));
        }
        let ones = vec![1u32; whos.len()];
        let slices: Vec<&[f64]> = probs.iter().map(|p| p.as_slice()).collect();
        // Safety: shared_costs is the config's table, alive as long as the engine.
        let mut counts = unsafe {
            allocate::allocate(
                &ones,
                &slices,
                &*self.shared_costs,
                self.overhead_ms(whos.len()),
                whos.len().max(self.batch_rows as usize) as u64,
            )
        };
        if self.plain_guard {
            // No draft where none pays; a probe after a growing run of plain rounds
            // (upstream rewrites each zero count with the probe's answer).
            for (w, c) in whos.iter().zip(counts.iter_mut()) {
                // Safety: the stream's depth state, under the crate invariants.
                unsafe {
                    let st = (*w.state).as_mut().unwrap();
                    st.rounds += 1;
                    if *c != 0 {
                        st.plain = Some(0);
                        st.wait = Some(self.k.depth_probe_every);
                    } else {
                        *c = u32::from(self.probe_plain(st));
                    }
                }
            }
            for (((w, m), c), o) in whos
                .iter()
                .zip(most.iter())
                .zip(counts.iter())
                .zip(out.iter_mut())
            {
                *o = self.head_depth(w, Some((*m).min(*c as i64)), probe)?;
            }
            return Ok(());
        }
        for (((w, m), c), o) in whos
            .iter()
            .zip(most.iter())
            .zip(counts.iter())
            .zip(out.iter_mut())
        {
            *o = self.head_depth(w, Some((*m).min(1.max(*c as i64))), probe)?;
        }
        Ok(())
    }

    /// A shared round's ms beyond its forward at this many streams (the nearest
    /// measured count, first seen on ties). Upstream: `overheadMs`.
    pub fn overhead_ms(&self, streams: usize) -> f64 {
        if self.overhead.is_empty() {
            return 8.0;
        }
        let mut best = self.overhead[0];
        for o in &self.overhead[1..] {
            if distance(o.streams, streams) < distance(best.streams, streams) {
                best = *o;
            }
        }
        best.ms
    }

    /// Upstream: `observeOverhead`.
    pub fn observe_overhead(&mut self, streams: u32, rows: u64, ms: f64) -> Result<(), Error> {
        // Safety: shared_costs is the config's table.
        let forward = unsafe { (*self.shared_costs).get(rows as i64) };
        let Some(forward) = forward else { return Ok(()) };
        let extra = if ms - forward > 0.0 { ms - forward } else { 0.0 };
        for o in self.overhead.iter_mut() {
            if o.streams == streams {
                o.ms = o.ms + 0.2 * (extra - o.ms);
                return Ok(());
            }
        }
        self.overhead.push(Overhead { streams, ms: extra });
        Ok(())
    }
}

fn distance(a: u32, b: usize) -> usize {
    (a as usize).abs_diff(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors depth.zig:259: depth picks the most tokens a ms and probes deeper.
    #[test]
    fn depth_picks_the_most_tokens_a_ms() {
        let mut fc = Table::new();
        for w in 1..5 {
            fc.put(w, 5.0 + w as f64);
        }
        let shared = Table::new();
        let mut rule = Rule::new(
            Constants::default(),
            &[0.9, 0.9, 0.9],
            3,
            &fc,
            &shared,
            0.5,
            false,
            false,
            8,
        );
        let mut st: Option<State> = None;
        let who = Who {
            state: &mut st,
            draft_room: 100,
            finished: false,
            forced: false,
            granted: None,
            stream: None,
        };
        assert_eq!(rule.depth(who).unwrap(), 3);
    }
}
