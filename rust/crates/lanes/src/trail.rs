//! What the round loop writes to the decision log: the recorder's events, field for
//! field (events.rs formats).
//!
//! Port of `zig/src/core/lanes/trail.zig`. All no-ops when the engine's log is
//! off (upstream: `event` checks `e.log`; the state builder too).

use crate::engine::Engine;
use crate::events::{f, Field, Value};
use crate::stream::{Mode, Stream};
use crate::table::Table;
use crate::Error;

/// Upstream: `trail.zig:13  pub fn str`.
pub fn str(s: &str) -> Value<'_> {
    Value::Str(s)
}

/// Upstream: `trail.zig:17  pub fn int`.
pub fn int(x: i64) -> Value<'static> {
    Value::Int(x)
}

/// Upstream: `trail.zig:21  pub fn event`.
pub fn event(e: &mut Engine<'_>, fields: &[Field<'_>]) -> Result<(), Error> {
    if let Some(log) = &mut e.log {
        log.add(fields);
    }
    Ok(())
}

/// Upstream: `trail.zig:25  pub fn pipe`.
///
/// Safety: `s` per the crate invariants (read only).
pub fn pipe(e: &mut Engine<'_>, s: *mut Stream<'_>, before: Option<Mode>, got: &[u32]) -> Result<(), Error> {
    // Safety: a read of the stream's own fields.
    let (id, mode) = unsafe { ((*s).id, (*s).mode) };
    event(
        e,
        &[
            f("ev", str("pipe")),
            f("stream", Value::Str(id)),
            f("mode", mode_name(before)),
            f("after", mode_name(mode)),
            f("got", Value::U32s(got)),
        ],
    )
}

/// Upstream: `trail.zig:29  pub fn modeName`.
pub fn mode_name(m: Option<Mode>) -> Value<'static> {
    match m {
        Some(x) => str(x.name()),
        None => Value::Null,
    }
}

/// Fill in queued draws' tokens where the recorder reads them: the end of an
/// admission or a step. Upstream: `trail.zig:34  pub fn resolve`.
pub fn resolve(e: &mut Engine<'_>) -> Result<(), Error> {
    if e.log.is_none() {
        return Ok(());
    }
    let unread: Vec<(usize, u64)> = e.unread.drain(..).map(|u| (u.event, u.handle)).collect();
    for (event, handle) in unread {
        let token = e.backend.read(handle)?;
        if let Some(log) = &mut e.log {
            log.set(event, "token", Value::Int(token as i64)).ok();
        }
    }
    Ok(())
}

/// Upstream: `trail.zig:40  pub fn finish`.
///
/// Safety: `s` per the crate invariants (read only).
pub fn finish(e: &mut Engine<'_>, s: *mut Stream<'_>) -> Result<(), Error> {
    // Safety: a read of the stream's own fields.
    let (id, reason, emitted, rounds, drafted, accepted, cache_len) = unsafe {
        (
            (*s).id,
            (*s).reason,
            (*s).emitted(),
            (*s).rounds,
            (*s).drafted,
            (*s).accepted,
            (*s).cache_len,
        )
    };
    event(
        e,
        &[
            f("ev", str("finish")),
            f("stream", Value::Str(id)),
            f("reason", str(reason.name())),
            f("emitted", Value::U32s(emitted)),
            f("rounds", int(rounds as i64)),
            f("drafted", int(drafted as i64)),
            f("accepted", int(accepted as i64)),
            f("cache_len", int(cache_len as i64)),
        ],
    )
}

/// A table as the recorder lists one: [key, bits] pairs. Upstream:
/// `trail.zig:44  pub fn table`.
pub fn table<'a>(arena: &'a bumpalo::Bump, t: &Table) -> Value<'a> {
    let mut rows: Vec<Value<'a>> = Vec::new();
    for (k, slot) in t.slots().iter().enumerate() {
        let Some(v) = *slot else { continue };
        let pair: &'a mut [Value<'a>] =
            arena.alloc_slice_copy(&[Value::Int(k as i64), crate::events::bits(v)]);
        rows.push(Value::List(pair));
    }
    Value::List(arena.alloc_slice_copy(&rows))
}

/// The loop's state after a step: the recorder's `state` event, field for field.
/// Upstream: `trail.zig:57  pub fn state`.
pub fn state(e: &mut Engine<'_>) -> Result<(), Error> {
    if e.log.is_none() {
        return Ok(());
    }
    let arena = e.arena();
    let overhead: &'static mut [Value<'static>] =
        arena.alloc_slice_copy(&e.rule.overhead.iter().map(|o| {
            Value::List(arena.alloc_slice_copy(&[
                Value::Int(o.streams as i64),
                crate::events::bits(o.ms),
            ]))
        }).collect::<Vec<_>>());
    let streams: &'static mut [Value<'static>] = arena.alloc_slice_copy(
        &e.live
            .iter()
            .map(|&s| {
                // Safety: a read of the stream's own fields, under the crate's
                // invariants.
                unsafe { stream_state(arena, s) }
            })
            .collect::<Vec<_>>(),
    );
    event(
        e,
        &[
            f("ev", str("state")),
            f("alone", Value::Bool(e.alone)),
            f("drafted", int(e.drafted as i64)),
            f("accepted", int(e.accepted as i64)),
            f("round_ms", table(arena, &e.rule.round_ms)),
            f("overhead_ms", Value::List(overhead)),
            f("shared_rounds", int(e.shared_rounds as i64)),
            f("streams", Value::List(streams)),
        ],
    )?;
    Ok(())
}

/// The state event's per-stream object (trail.zig:67-89, field for field).
///
/// Safety: `s` per the crate invariants (read only).
unsafe fn stream_state<'a>(arena: &'a bumpalo::Bump, s: *mut Stream<'_>) -> Value<'a> {
    unsafe {
        let x = &*s;
        let id = arena.alloc_str(x.id);
        let emitted_len = x.emitted().len();
        let pending: &'a [u32] = match x.pending {
            Some(p) => arena.alloc_slice_copy(&[p]),
            None => &[],
        };
        let d = match &x.depth {
            Some(st) => {
                let p: &'a [u64] =
                    arena.alloc_slice_copy(&st.p.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                let ms = match &st.ms {
                    Some(t) => table(arena, t),
                    None => Value::Null,
                };
                Value::Obj(arena.alloc_slice_copy(&[
                    f("p", Value::U64s(p)),
                    f("rounds", int(st.rounds as i64)),
                    f("plain", st.plain.map(|v| int(v as i64)).unwrap_or(Value::Null)),
                    f("wait", st.wait.map(|v| int(v as i64)).unwrap_or(Value::Null)),
                    f("ms", ms),
                ]))
            }
            None => Value::Null,
        };
        let proposer = match x.proposer.as_ref() {
            Some(p) => match p.state_fields() {
                Some(fields) => Value::Obj(arena.alloc_slice_copy(
                    &fields
                        .iter()
                        .map(|(k, v)| Field { k: arena.alloc_str(k), v: arena_value(arena, v) })
                        .collect::<Vec<_>>(),
                )),
                None => Value::Null,
            },
            None => Value::Null,
        };
        Value::Obj(arena.alloc_slice_copy(&[
            f("id", Value::Str(id)),
            f("cache_len", int(x.cache_len as i64)),
            f("emitted", int(emitted_len as i64)),
            f("pending", Value::U32s(pending)),
            f("rounds", int(x.rounds as i64)),
            f("drafted", int(x.drafted as i64)),
            f("accepted", int(x.accepted as i64)),
            f("force", Value::U32s(arena.alloc_slice_copy(&x.force))),
            f("depth", d),
            f("mode", mode_name(x.mode)),
            f("copy_width", x.copy_width.map(|v| int(v as i64)).unwrap_or(Value::Null)),
            f("served", x.served.map(|v| int(v as i64)).unwrap_or(Value::Null)),
            f("granted", x.granted.map(|v| int(v as i64)).unwrap_or(Value::Null)),
            f("next", x.next.as_ref().map(|h| int(h.count as i64)).unwrap_or(Value::Null)),
            f("inflight", Value::Bool(x.inflight.is_some())),
            f("proposer", proposer),
        ]))
    }
}

/// A proposer state value copied into the arena, so the state object's fields share
/// one lifetime (the proposer's slices are the only borrowed kind it gives).
fn arena_value<'a>(arena: &'a bumpalo::Bump, v: &Value<'_>) -> Value<'a> {
    match v {
        Value::Str(s) => Value::Str(arena.alloc_str(s)),
        Value::U32s(x) => Value::U32s(arena.alloc_slice_copy(x)),
        Value::I32s(x) => Value::I32s(arena.alloc_slice_copy(x)),
        Value::I64s(x) => Value::I64s(arena.alloc_slice_copy(x)),
        Value::U64s(x) => Value::U64s(arena.alloc_slice_copy(x)),
        Value::Null => Value::Null,
        Value::Bool(b) => Value::Bool(*b),
        Value::Int(i) => Value::Int(*i),
        Value::Uint(u) => Value::Uint(*u),
        Value::List(_) | Value::Obj(_) => Value::Null,
    }
}
