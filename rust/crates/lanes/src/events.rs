//! The decision log in the Python recorder's format: one JSON line an event, keys
//! sorted, floats as their bits.
//!
//! Port of `zig/src/core/lanes/events.zig`. The exact bytes matter: the recorder
//! (Python) writes `json.dumps(sort_keys=True)` with floats as their IEEE bits,
//! and the trail tests pin whole lines.

use std::fmt::Write as _;

/// The log has no such field to fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoSuchField;

/// Upstream: `events.zig:5  pub const Value = union(enum) { ... }`.
#[derive(Clone, Copy, Debug)]
pub enum Value<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Str(&'a str),
    U32s(&'a [u32]),
    I32s(&'a [i32]),
    I64s(&'a [i64]),
    U64s(&'a [u64]),
    List(&'a [Value<'a>]),
    Obj(&'a [Field<'a>]),
}

/// Upstream: `events.zig:19  pub const Field = struct { k, v }`.
#[derive(Clone, Copy, Debug)]
pub struct Field<'a> {
    pub k: &'a str,
    pub v: Value<'a>,
}

/// Upstream: `events.zig:21  pub fn f(k, v)`.
pub fn f<'a>(k: &'a str, v: Value<'a>) -> Field<'a> {
    Field { k, v }
}

/// A double as its IEEE bits, the way the recorder writes floats.
/// Upstream: `events.zig:26  pub fn bits(x)`.
pub fn bits(x: f64) -> Value<'static> {
    Value::Uint(x.to_bits())
}

/// A deep copy of a field list, owned by the log's arena; the trail builds events
/// from borrowed slices.
/// Deep-copy a field list into `out`; every borrowed byte — keys included — is
/// copied into the bump, so the copy owns its whole graph and lives as long as the
/// bump does. The transmute at the end re-lifetimes the copied slice to match the
/// `Log`'s storage (the log owns the bump; nothing else reaches these fields).
// The returned slice is a fresh arena allocation (never a view of `fields`); the
// transmute inside copies the input once for `alloc_slice_copy`.
#[allow(clippy::mut_from_ref)]
pub(crate) unsafe fn dupe_fields(out: &'static bumpalo::Bump, fields: &[Field<'_>]) -> &'static mut [Field<'static>] {
    // Safety: the transmute only re-lifetimes the input for the copy call; the
    // bytes are read once and the output owns its own graph (keys re-allocated
    // below, values deep-copied by dupe_value).
    let copy: &'static mut [Field<'static>] = unsafe {
        let fields: &[Field<'static>] = std::mem::transmute(fields);
        out.alloc_slice_copy(fields)
    };
    for field in copy.iter_mut() {
        field.k = out.alloc_str(field.k);
        field.v = dupe_value(out, &field.v);
    }
    copy
}

fn dupe_value<'out>(out: &'static bumpalo::Bump, v: &Value<'_>) -> Value<'out> {
    match v {
        Value::Str(s) => Value::Str(out.alloc_str(s)),
        Value::U32s(s) => Value::U32s(out.alloc_slice_copy(s)),
        Value::I32s(s) => Value::I32s(out.alloc_slice_copy(s)),
        Value::I64s(s) => Value::I64s(out.alloc_slice_copy(s)),
        Value::U64s(s) => Value::U64s(out.alloc_slice_copy(s)),
        Value::List(s) => {
            let copy = out.alloc_slice_copy(s.as_ref());
            let copied: Vec<Value<'out>> = copy
                .iter()
                .map(|x| dupe_value(out, x))
                .collect();
            Value::List(out.alloc_slice_copy(&copied))
        }
        Value::Obj(s) => Value::Obj(unsafe { dupe_fields(out, s) }),
        Value::Null => Value::Null,
        Value::Bool(b) => Value::Bool(*b),
        Value::Int(i) => Value::Int(*i),
        Value::Uint(u) => Value::Uint(*u),
    }
}

/// The log the round loop appends to, when the recorder is on.
///
/// Upstream: `events.zig:30  pub const Log = struct { arena, events }` — a deep
/// copy per event, index-addressable so the loop can fill a token it reads later.
pub struct Log {
    arena: bumpalo::Bump,
    events: Vec<&'static mut [Field<'static>]>,
}

impl Default for Log {
    fn default() -> Self {
        Log::new()
    }
}

impl Log {
    pub fn new() -> Log {
        Log {
            arena: bumpalo::Bump::new(),
            events: Vec::new(),
        }
    }

    /// The log's arena, unbounded: the log owns it, its events only read through
    /// the log, and the log outlives every line it prints.
    ///
    /// Safety: sound because nothing else can reach the arena's allocations; the
    /// accessor mirrors upstream's separate allocator, whose product slices
    /// outlive any one borrow of the log.
    fn arena(&self) -> &'static bumpalo::Bump {
        unsafe { &*(&self.arena as *const bumpalo::Bump) }
    }

    /// Record an event (deep-copied); its index lets the loop fill a token it
    /// reads later. Upstream: `events.zig:43  pub fn add`.
    pub fn add(&mut self, fields: &[Field<'_>]) -> usize {
        // Safety: the copies live in the log's own arena and only read through its
        // events; both live as long as the log.
        let copy = unsafe { dupe_fields(self.arena(), fields) };
        self.events.push(copy);
        self.events.len() - 1
    }

    /// Upstream: `events.zig:51  pub fn set`.
    pub fn set(&mut self, event: usize, k: &str, v: Value<'_>) -> Result<(), NoSuchField> {
        // Safety: as add — the copy lives in the log's own arena.
        let v = unsafe {
            std::mem::transmute::<Value<'_>, Value<'static>>(dupe_value(self.arena(), &v))
        };
        for x in self.events[event].iter_mut() {
            if x.k == k {
                x.v = v;
                return Ok(());
            }
        }
        Err(NoSuchField)
    }

    /// One event as the recorder's JSON line (no newline). Upstream:
    /// `events.zig:62  pub fn line`.
    pub fn line(&self, event: usize) -> String {
        let mut out = String::new();
        write_obj(self.events[event].iter(), &mut out);
        out
    }
}

fn write_obj<'a>(fields: impl Iterator<Item = &'a Field<'a>>, out: &mut String) {
    let mut sorted: Vec<&Field<'_>> = fields.collect();
    sorted.sort_by(|a, b| a.k.cmp(b.k));
    out.push('{');
    for (i, x) in sorted.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_str(out, x.k);
        out.push(':');
        write_value(&x.v, out);
    }
    out.push('}');
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 10 || ((c as u32) >= 11 && (c as u32) <= 31) => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_ints<T: std::fmt::Display>(out: &mut String, s: &[T]) {
    out.push('[');
    for (i, x) in s.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{}", x);
    }
    out.push(']');
}

fn write_value(v: &Value<'_>, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(x) => {
            let _ = write!(out, "{}", x);
        }
        Value::Uint(x) => {
            let _ = write!(out, "{}", x);
        }
        Value::Str(s) => write_str(out, s),
        Value::U32s(s) => write_ints(out, s),
        Value::I32s(s) => write_ints(out, s),
        Value::I64s(s) => write_ints(out, s),
        Value::U64s(s) => write_ints(out, s),
        Value::List(s) => {
            out.push('[');
            for (i, x) in s.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(x, out);
            }
            out.push(']');
        }
        Value::Obj(s) => write_obj(s.iter(), out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors events.zig:153: lines sort keys like json.dumps(sort_keys=True),
    /// floats as bits, arrays as Python lists.
    #[test]
    fn lines_sort_keys_like_json_dumps() {
        let mut log = Log::new();
        let i = log.add(&[
            f("stream", Value::Str("g")),
            f("ev", Value::Str("plan")),
            f("n", Value::Int(2)),
            f("tree", Value::Bool(false)),
            f("got", Value::U32s(&[1, 2])),
            f("cut", Value::Null),
        ]);
        assert_eq!(
            log.line(i),
            "{\"cut\":null,\"ev\":\"plan\",\"got\":[1,2],\"n\":2,\"stream\":\"g\",\"tree\":false}"
        );
    }

    /// A float writes as its bits, and `set` fills a queued draw's token.
    #[test]
    fn floats_write_as_bits_and_set_fills_later() {
        let mut log = Log::new();
        let i = log.add(&[f("ms", bits(1.5)), f("token", Value::Null)]);
        assert_eq!(log.line(i), format!("{{\"ms\":{},\"token\":null}}", 1.5f64.to_bits()));
        log.set(i, "token", Value::Int(7)).unwrap();
        assert_eq!(log.line(i), format!("{{\"ms\":{},\"token\":7}}", 1.5f64.to_bits()));
    }
}
