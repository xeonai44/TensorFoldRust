//! A lone drafted stream decoded as `tensorfold run` decodes: the
//! confidence depth rule, copies, the graphs.
//!
//! Port of `zig/src/families/nemotron/cuda_lone.zig` (45 lines).

use tensorfold_core::draft_depth::DepthRule;
use tensorfold_lanes::stream::Stream;

use crate::cuda_drafts::depth_rule;
use crate::cuda_lanes::Cuda;
use crate::error::Error;

/// The server's engine thread between rounds: told when tokens land,
/// asked whether to hand the stream to the lane core. Upstream:
/// `Hooks` (cuda_lone.zig:9) — the port's pair of boxed callbacks.
pub struct Hooks<'h> {
    pub ctx: *mut core::ffi::c_void,
    pub committed: &'h mut dyn FnMut(*mut core::ffi::c_void),
    pub yield_now: &'h mut dyn FnMut(*mut core::ffi::c_void) -> bool,
}

/// Prefill `s` and decode it until it finishes (false) or `yield`
/// hands it to the lane core (true). Upstream: `run`
/// (cuda_lone.zig:16).
pub fn run(b: &mut Cuda<'_, '_>, s: &mut Stream<'_>, hooks: Hooks<'_>) -> Result<bool, Error> {
    let h = match b.e().head.as_ref() {
        Some(h) => h,
        None => return Err(Error::NoMtpHead),
    };
    let _ = h;
    if !s.drafts {
        return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // DraftedOnly
    }
    let measured = match &b.measured {
        Some(c) => *c,
        None => return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)), // CostsNotMeasured
    };
    let mut rule: DepthRule = depth_rule(measured);
    // the prompt pass, its first token committed; a cancel or failure
    // releases the stream
    b.opening_lone(s)?;
    (hooks.committed)(hooks.ctx);
    if s.finished {
        return Ok(false);
    }
    let mut rounds = crate::cuda_drafts::Rounds::init(b.e_mut(), &mut rule, s.prompt(), s.emitted(), None)?;
    loop {
        let last = s.context[s.context.len() - 1];
        let (sampled, accepted) = rounds.verify(last)?;
        let kept = accepted + 1;
        // lands all `kept` unless the stream finishes inside them
        s.commit(&sampled[..kept], &[])
            .map_err(|e| Error::Backend(e.to_string()))?;
        s.rounds += 1;
        s.drafted += rounds.proposal_len as u64;
        s.accepted += accepted as u64;
        (hooks.committed)(hooks.ctx);
        if s.finished {
            return Ok(false);
        }
        let handing = (hooks.yield_now)(hooks.ctx);
        rounds.keep(&sampled, kept, !handing)?;
        if !handing {
            continue;
        }
        b.hand_over(s, kept)?;
        return Ok(true);
    }
}

impl<'a, 'd> Cuda<'a, 'd> {
    /// The lone opening: upstream's `be.opening(gpa, s)` over the
    /// backend trait — the prompt pass and the first token, committed,
    /// a cancel or failure releasing the stream.
    fn opening_lone(&mut self, s: &mut Stream<'_>) -> Result<(), Error> {
        use tensorfold_lanes::backend::Backend;
        let sptr = s as *mut Stream<'_>;
        let position = s.prompt_len as u64;
        Backend::prefill(self, sptr).map_err(|e| Error::Backend(e.to_string()))?;
        let handle =
            Backend::first(self, sptr, position).map_err(|e| Error::Backend(e.to_string()))?;
        let token = Backend::read(self, handle).map_err(|e| Error::Backend(e.to_string()))?;
        s.commit(&[token], &[])
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok(())
    }
}
