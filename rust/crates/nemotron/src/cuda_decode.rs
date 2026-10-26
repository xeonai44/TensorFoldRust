//! decode.prefill + serial_decode on the Zig engine: the same rounds,
//! the same tokens, timed the same way.
//!
//! Port of `zig/src/families/nemotron/cuda_decode.zig` (71 lines).

use std::time::Instant;

use crate::cuda_drafts::{self as drafts, Drafter, Stats};
use crate::cuda_engine::{Engine, LOOKAHEAD};
use crate::cuda_state::PREFILL_ROWS;
use crate::error::Error;

/// Upstream: `Options` (cuda_decode.zig:10).
#[derive(Clone, Copy)]
pub struct Options {
    pub stop_eos: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { stop_eos: true }
    }
}

/// Upstream: `Result` (cuda_decode.zig:12).
pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
}

/// `count` tokens (the first from the prompt's last row), stopping
/// after an end token when `stop_eos`. Upstream: `generate`
/// (cuda_decode.zig:22).
pub fn generate(
    e: &mut Engine<'_, '_>,
    drafter: Option<&mut Drafter>,
    prompt: &[u32],
    count: usize,
    o: Options,
) -> Result<DecodeResult, Error> {
    let mut out: Vec<u32> = Vec::new();
    let has_head = e.head.is_some();
    let t0 = Instant::now();
    out.push(e.prefill(prompt, has_head)?);
    let prefill_s = t0.elapsed().as_secs_f64();
    e.stream.synchronize()?;
    let t1 = Instant::now();
    let mut st = Stats::default();
    if out.len() < count && !(o.stop_eos && e.c.is_eos(out[0])) {
        if let Some(d) = drafter {
            let last = (prompt.len() - 1) % PREFILL_ROWS;
            let hidden = e.b.p_hidden + last as u64 * e.c.hidden as u64 * 2;
            st = drafts::decode(e, &mut d.rule, prompt, &mut out, count, o.stop_eos, hidden)?;
        } else if e.serial.is_some() {
            st.rounds = serial_graphs(e, &mut out, count, o.stop_eos)?;
        } else {
            while out.len() < count && !(o.stop_eos && e.c.is_eos(out[out.len() - 1])) {
                out.push(e.step(out[out.len() - 1])?);
                st.rounds += 1;
            }
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    e.stream.synchronize()?;
    Ok(DecodeResult {
        tokens: out,
        prefill_seconds: prefill_s,
        decode_seconds: decode_s,
        rounds: st.rounds,
        drafted: st.drafted,
        accepted: st.accepted,
    })
}

/// serial_decode as one graph replay a round, queued `lookahead` deep:
/// the device feeds each token to the next round. Upstream:
/// `serialGraphs` (cuda_decode.zig:49).
fn serial_graphs(
    e: &mut Engine<'_, '_>,
    out: &mut Vec<u32>,
    count: usize,
    stop_eos: bool,
) -> Result<usize, Error> {
    e.upload(out[out.len() - 1])?;
    let base = e.pos;
    let total = count - out.len();
    let mut launched: usize = 0;
    let mut read: usize = 0;
    while read < total {
        while launched < total && launched - read < LOOKAHEAD {
            e.serial.as_ref().unwrap().launch_on(&e.stream)?;
            e.done[launched % LOOKAHEAD].record(&e.stream)?;
            launched += 1;
        }
        e.done[read % LOOKAHEAD].synchronize()?;
        let tok = e.written(base + read + 1);
        out.push(tok);
        read += 1;
        if stop_eos && e.c.is_eos(tok) {
            break;
        }
    }
    e.pos = base + launched;
    e.parity ^= launched & 1;
    if launched > 0 {
        e.prev_keep = 1;
    }
    Ok(read)
}
