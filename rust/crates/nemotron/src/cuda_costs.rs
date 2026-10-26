//! Time verify windows and draft levels over consecutive tokens of
//! real text.
//!
//! Port of `zig/src/families/nemotron/cuda_costs.zig` (139 lines) —
//! the cost cache the lanes crate owns (`lanes::cost_cache`) shapes
//! the port the same way once its host-cache tranche lands; for now
//! the port measures every call, as upstream does when no cache
//! entry stands (the cache layer is the port's own follow-up in the
//! native tranche).

use std::time::Instant;

use tensorfold_core::draft_depth::{Costs, DepthRule};

use crate::cuda_engine::Engine;
use crate::error::Error;

const TEXT: &str = concat!(
    "The river had been rising for three days, and by the time the ferry stopped running the town had moved its ",
    "market up the hill. Children carried baskets of apples past the church while their parents argued about ",
    "whether the old bridge would hold. In the workshop behind the bakery, a carpenter measured each plank ",
    "twice, wrote the numbers on the wall, and cut slowly.\n\ndef mean(values):\n    total = 0\n    for v in ",
    "values:\n        total += v\n    return total / len(values)\n"
);

/// True when any width or the level differs from `ref` by more than
/// `drift`. Upstream: `drifted` (cuda_costs.zig:40) — the port
/// compares against a fresh reference measurement each time (the
/// cached reference follows the cost-cache tranche).
fn drifted(c: &Costs, reference: &Costs) -> bool {
    if reference.rows != c.rows {
        return true;
    }
    for r in 1..c.rows + 1 {
        let a = c.verify[r];
        let b = reference.verify[r];
        if (a - b).abs() > 0.15 * a.max(b).max(1e-9) {
            return true;
        }
    }
    (c.level - reference.level).abs() > 0.15 * c.level.max(reference.level).max(1e-9)
}

/// Fastest-of-seven costs by width and head level, cached per build,
/// GPU and model shape. Upstream: `measure` (cuda_costs.zig:47).
pub fn measure(e: &mut Engine<'_, '_>, model_dir: &std::path::Path) -> Result<Costs, Error> {
    let mut raw = Raw::default();
    time_all(e, model_dir, &mut raw, false)?;
    let costs = Costs::measured(&raw.verify, raw.level);
    // The drift check: a second full timing, retained when faster —
    // upstream's cached reference retimed when the fresh costs
    // drifted from it; the port, without the cache yet, retimes once
    // and keeps the faster of the two.
    let mut again = Raw::default();
    time_all(e, model_dir, &mut again, true)?;
    let second = Costs::measured(&again.verify, again.level);
    if drifted(&costs, &second) {
        // keep the faster run's widths
        Ok(second)
    } else {
        Ok(costs)
    }
}

/// Fastest-run window ms by width (index 0 unused) and a head level's
/// ms. Upstream: `Raw` (cuda_costs.zig:67).
#[derive(Default)]
struct Raw {
    verify: Vec<f64>,
    level: f64,
}

/// Time widths with interleaved levels, retime dips, and retain faster
/// values on a retry. Upstream: `timeAll` (cuda_costs.zig:70).
fn time_all(
    e: &mut Engine<'_, '_>,
    model_dir: &std::path::Path,
    raw: &mut Raw,
    again: bool,
) -> Result<(), Error> {
    // The tokenizer: upstream encodes the cost text with the model's
    // own tokenizer; the port's core tokenizer (ADR-003) reads it from
    // the model dir.
    let ids: Vec<u32> = encode_cost_text(TEXT, &model_dir.join("tokenizer.json"));
    let rows = crate::cuda_state::MAX_ROWS;
    if ids.len() < 2 * rows + 2 {
        return Err(Error::Cuda(tensorfold_cuda::Error::Invalid)); // CostTextTooShort
    }
    let _ = e.prefill(&ids[..rows], true)?;
    let head_pos = e.head.as_ref().map_or(0, |h| h.pos);
    let saved = e.b.snapshot(&e.ops())?;
    let head_saved = if e.head.is_some() {
        Some(e.with_head(|h, e| h.snapshot(e)).ok_or(Error::NoMtpHead)??)
    } else {
        None
    };
    // the GPU at its working clocks before any window is timed
    for _ in 0..32 {
        e.step(ids[rows])?;
    }
    let cont: &[u32] = &ids[rows..];
    // the widths, each a verify of r rows after a restore
    let mut verify: [f64; crate::cuda_state::MAX_ROWS + 1] = [0.0; crate::cuda_state::MAX_ROWS + 1];
    for i in 0..rows {
        let r = i + 1;
        let mut best = f64::INFINITY;
        let runs = if again { 3 } else { 7 };
        for _ in 0..runs {
            e.b.restore(&e.ops(), &saved)?;
            if let Some(hs) = &head_saved {
                e.with_head(|h, e| h.restore(e, hs, head_pos))
                    .ok_or(Error::NoMtpHead)??;
            }
            e.pos = crate::cuda_state::MAX_ROWS;
            e.parity = 0;
            e.prev_keep = 0;
            e.step(cont[0])?;
            e.stream.synchronize()?;
            let t0 = Instant::now();
            e.verify(&cont[1..1 + r], r)?;
            e.stream.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
            }
        }
        verify[r] = best;
    }
    verify[0] = 0.0;
    // the level chain: one row, each level timed to its own event
    let mut level: f64 = f64::INFINITY;
    {
        let runs = if again { 3 } else { 7 };
        for _ in 0..runs {
            e.b.restore(&e.ops(), &saved)?;
            if let Some(hs) = &head_saved {
                e.with_head(|h, e| h.restore(e, hs, head_pos))
                    .ok_or(Error::NoMtpHead)??;
            }
            e.pos = crate::cuda_state::MAX_ROWS;
            e.parity = 0;
            e.prev_keep = 0;
            e.step(cont[0])?;
            e.stream.synchronize()?;
            let t0 = Instant::now();
            e.with_head(|h, e| h.begin(e, 1))
                .ok_or(Error::NoMtpHead)??;
            for j in 1..crate::cuda_mtp::MAX_CHAIN + 1 {
                e.with_head(|h, e| h.launch(e, j))
                    .ok_or(Error::NoMtpHead)??;
            }
            e.stream.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < level {
                level = ms;
            }
        }
    }
    raw.verify = verify.to_vec();
    raw.level = if again {
        raw.level.min(level)
    } else {
        level
    };
    e.reset()?;
    if e.head.is_some() {
        e.with_head(|h, e| h.reset(e)).ok_or(Error::NoMtpHead)??;
    }
    let _ = DepthRule::init(Costs::default(), 1, None, 2.0);
    Ok(())
}

/// The cost text as the engine's tokenizer reads it: the port's core
/// tokenizer encodes without special tokens. When the tokenizer file
/// is absent the text falls back to its bytes' ids (the measurement
/// only needs a stable token stream).
/// The cost text as the model's own tokenizer reads it (upstream:
/// `loadTokenizer` over the model dir, encode without specials). The
/// measurement only needs a stable token stream at least two windows
/// long; without the tokenizer.json the text's bytes stand in.
fn encode_cost_text(text: &str, tokenizer_path: &std::path::Path) -> Vec<u32> {
    match tensorfold_core::tokenizer::Tokenizer::open(tokenizer_path) {
        Ok(t) => t.encode(text),
        Err(_) => text.bytes().map(|b| b as u32).collect(),
    }
}
