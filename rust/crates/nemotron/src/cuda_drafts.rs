//! decode.draft_decode: MTP chains (or copied chains) verified in one
//! window a round; kept drafts equal serial tokens.
//!
//! Port of `zig/src/families/nemotron/cuda_drafts.zig` (142 lines) —
//! the head through the engine's `with_head` bridge, the rule read by
//! reference as upstream's `*DepthRule` is.

use tensorfold_core::copy_index::CopyIndex;
use tensorfold_core::draft_depth::{Costs, DepthRule, MAX_ROWS};

use crate::cuda_engine::Engine;
use crate::cuda_mtp::MAX_CHAIN as MTP_MAX_CHAIN;
use crate::error::Error;

/// Upstream: `Stats` (cuda_drafts.zig:10).
#[derive(Clone, Copy, Default)]
pub struct Stats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
}

/// Start a round's head work: an absorb for a copied chain, else level
/// 1 (the rule reads the later levels). Upstream: `queue`
/// (cuda_drafts.zig:13).
fn queue(e: &mut Engine<'_, '_>, keep: usize, copied: bool) -> Result<(), Error> {
    e.with_head(|h, e| h.begin(e, keep))
        .ok_or(Error::NoMtpHead)??;
    e.with_head(|h, e| h.launch(e, if copied { 0 } else { 1 }))
        .ok_or(Error::NoMtpHead)??;
    Ok(())
}

/// The drafts this round verifies: the rule reads each level's
/// confidence before drafting the next. Upstream: `depth`
/// (cuda_drafts.zig:19).
fn depth(e: &mut Engine<'_, '_>, rule: &DepthRule) -> Result<usize, Error> {
    let mut run: f64 = 1.0;
    let mut n: usize = 0;
    for j in 1..=MTP_MAX_CHAIN {
        if j > 1 {
            e.with_head(|h, e| h.launch(e, j)).ok_or(Error::NoMtpHead)??;
        }
        let conf = e
            .with_head(|h, _| h.confidence(j))
            .ok_or(Error::NoMtpHead)??;
        run *= conf as f64;
        if !rule.keep(j, run) {
            break;
        }
        n = j;
        if !rule.more(j, run) {
            break;
        }
    }
    Ok(n)
}

/// A drafted decode's rounds over a token context: each verifies a
/// copied chain or the head's chain in one window. Upstream: `Rounds`
/// (cuda_drafts.zig:34).
pub struct Rounds<'r> {
    /// The engine, as `with_head`'s bridge documents it (boxed by its
    /// owner, alive for the rounds); the raw pointer erases its two
    /// lifetime params (the rounds live inside their scope).
    e: *mut Engine<'static, 'static>,
    rule: &'r mut DepthRule,
    index: CopyIndex,
    cbuf: [u32; MTP_MAX_CHAIN],
    ids: [u32; MTP_MAX_CHAIN + 1],
    copied: Vec<u32>,
    pub proposal_len: usize,
    copy_round: bool,
    pub st: Stats,
}

impl<'a> Rounds<'a> {
    /// The head reads the prompt's last hidden row and final output
    /// token. Upstream: `init` (cuda_drafts.zig:47).
    pub fn init<'r>(
        e: &mut Engine<'_, '_>,
        rule: &'r mut DepthRule,
        prompt: &[u32],
        out: &[u32],
        last_hidden: Option<u64>,
    ) -> Result<Rounds<'r>, Error> {
        let mut index = CopyIndex::init(prompt);
        index.extend(out);
        if let Some(x) = last_hidden {
            e.ops().copy(e.b.hidden, x, e.c.hidden * 2)?;
        }
        e.ops().fill32(e.b.sampled, out[out.len() - 1], 1)?;
        // SAFETY: the rounds' bridge — the raw pointer's erased
        // lifetimes stand inside the rounds' scope.
        let e_ptr = (e as *mut Engine<'_, '_>).cast::<Engine<'static, 'static>>();
        let r_rule: &'r mut DepthRule = rule;
        let mut r = Rounds {
            e: e_ptr,
            rule: r_rule,
            index,
            cbuf: [0; MTP_MAX_CHAIN],
            ids: [0; MTP_MAX_CHAIN + 1],
            copied: Vec::new(),
            proposal_len: 0,
            copy_round: false,
            st: Stats::default(),
        };
        let copied: Vec<u32> = r.index.chain(&mut r.cbuf).to_vec();
        let copied_len = copied.len();
        r.copied = copied;
        // SAFETY: the rounds' bridge (with_head's invariants).
        queue(unsafe { &mut *r.e }, 1, copied_len > 0)?;
        Ok(r)
    }

    /// One window after `last`: the copied chain or the head's drafts,
    /// verified; its drawn tokens and drafts accepted. Upstream:
    /// `verify` (cuda_drafts.zig:63).
    pub fn verify(&mut self, last: u32) -> Result<(Vec<u32>, usize), Error> {
        // SAFETY: the rounds' bridge (with_head's invariants).
        // SAFETY: the rounds' bridge.
        let e: &mut Engine<'_, '_> = unsafe { &mut *self.e.cast::<Engine<'_, '_>>() };
        self.copy_round = !self.copied.is_empty();
        if self.copy_round {
            self.ids[0] = last;
            self.ids[1..1 + self.copied.len()].copy_from_slice(&self.copied);
            e.verify(&self.ids[..1 + self.copied.len()], 1 + self.copied.len())?;
            self.proposal_len = self.copied.len();
        } else {
            let n = depth(e, self.rule)?;
            e.verify(&[last], 1 + n)?;
            let drafts: Vec<u32> = e
                .with_head(|h, _| h.drafts().to_vec())
                .ok_or(Error::NoMtpHead)?
                .into_iter()
                .take(n)
                .collect();
            self.proposal_len = drafts.len();
        }
        let sampled = e.tokens()?;
        let mut accepted: usize = 0;
        let proposal: &[u32] = if self.copy_round {
            &self.copied
        } else {
            &[]
        };
        while accepted < proposal.len() && proposal[accepted] == sampled[accepted] {
            accepted += 1;
        }
        Ok((sampled, accepted))
    }

    /// Keep leading rows and their sampled tokens, then queue head work
    /// when more tokens are needed. Upstream: `keep`
    /// (cuda_drafts.zig:83).
    pub fn keep(&mut self, sampled: &[u32], n: usize, more: bool) -> Result<(), Error> {
        // SAFETY: the rounds' bridge.
        // SAFETY: the rounds' bridge.
        let e: &mut Engine<'_, '_> = unsafe { &mut *self.e.cast::<Engine<'_, '_>>() };
        e.commit(n)?;
        let levels = if self.copy_round {
            0
        } else {
            e.head.as_ref().map_or(0, |h| h.levels)
        };
        self.rule.done(n, 1 + self.proposal_len, levels);
        self.index.extend(&sampled[..n]);
        self.st.rounds += 1;
        self.st.drafted += self.proposal_len;
        self.st.accepted += n - 1;
        if !more {
            return Ok(());
        }
        let copied: Vec<u32> = self.index.chain(&mut self.cbuf).to_vec();
        let copied_len = copied.len();
        self.copied = copied;
        queue(e, n, copied_len > 0)?;
        Ok(())
    }
}

/// `count` tokens in `out` (its first the prompt's pending token),
/// drafting from the prompt's last hidden row. Upstream: `decode`
/// (cuda_drafts.zig:97).
pub fn decode(
    e: &mut Engine<'_, '_>,
    rule: &mut DepthRule,
    prompt: &[u32],
    out: &mut Vec<u32>,
    count: usize,
    stop_eos: bool,
    last_hidden: u64,
) -> Result<Stats, Error> {
    let mut r = Rounds::init(e, rule, prompt, out, Some(last_hidden))?;
    while out.len() < count && !(stop_eos && e.c.is_eos(out[out.len() - 1])) {
        let (sampled, mut accepted) = r.verify(out[out.len() - 1])?;
        if stop_eos {
            for (j, t) in sampled[..accepted].iter().enumerate() {
                if e.c.is_eos(*t) {
                    accepted = j;
                    break;
                }
            }
        }
        accepted = accepted.min(count - out.len() - 1);
        out.extend_from_slice(&sampled[..accepted + 1]);
        let more = out.len() < count && !(stop_eos && e.c.is_eos(out[out.len() - 1]));
        r.keep(&sampled, accepted + 1, more)?;
    }
    Ok(r.st)
}

/// The MTP head and the measured-cost depth rule (app._rules: greedy
/// calibration power 2, every draft depth). Upstream: `Drafter`
/// (cuda_drafts.zig:115) — the head lives in the engine's `with_head`
/// bridge; the rule here.
pub struct Drafter {
    pub rule: DepthRule,
}

impl Drafter {
    /// `costs`: window ms at 1..16 rows then a level's ms, as another
    /// run measured them; None measures them here. Upstream: `init`
    /// (cuda_drafts.zig:120).
    pub fn init(
        e: &mut Engine<'_, '_>,
        _model_dir: &std::path::Path,
        graphs: bool,
        costs: Option<&[f64]>,
    ) -> Result<Drafter, Error> {
        if graphs {
            e.with_head(|h, e| h.capture(e))
                .ok_or(Error::NoMtpHead)??;
        }
        let c: Costs = match costs {
            Some(given) => {
                if given.len() != 17 {
                    return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // CostsNeedSeventeenValues
                }
                let mut c = Costs {
                    verify: [0.0; MAX_ROWS + 1],
                    rows: 16,
                    level: given[16],
                };
                c.verify[1..17].copy_from_slice(&given[0..16]);
                c
            }
            None => crate::cuda_costs::measure(e, _model_dir)?,
        };
        Ok(Drafter {
            rule: depth_rule(c),
        })
    }
}

/// The depth rule `run` and a lone served stream draft by: greedy
/// calibration power 2, every draft depth. Upstream: `depthRule`
/// (cuda_drafts.zig:140).
pub fn depth_rule(c: Costs) -> DepthRule {
    DepthRule::init(c, MTP_MAX_CHAIN, None, 2.0)
}
