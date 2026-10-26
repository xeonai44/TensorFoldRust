//! Exact prompt reuse: states are keyed by tokens and canonical decoded
//! intervals through their position.
//!
//! Port of `zig/src/core/prompt_cache.zig` (528 lines). The learned-disk
//! layer (`prompt_imprint.zig`, `prompt_learn.zig`, upstream's `--learn`)
//! and the peer-disk coordination ride later, with the store's optional
//! hooks the shape they fill.

use crate::cache_modes::{self, Span};
use crate::learned_dirs;
use crate::prompt_learn::note;
use crate::prompt_imprint::Imprint;
use std::any::Any;

/// A family's copy of one state. Upstream: `Saved` (prompt_cache.zig:6) —
/// an opaque pointer the family owns; here an owned box the family's
/// `drop_saved` frees.
pub type Saved = Box<dyn Any + Send>;

/// Why a copy failed. Upstream: the error set of the vtable's functions.
#[derive(Debug)]
pub enum StoreError {
    /// The live state is not at the mark.
    NotAtMark,
    /// The copy itself failed.
    CopyFailed,
    /// A learned state's file write failed (prompt_learn's `write`).
    WriteFailed,
    /// A learned state's file read failed.
    ReadFailed,
    /// A learned state's deletion failed.
    DeleteFailed,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::NotAtMark => write!(f, "NotAtMark"),
            StoreError::CopyFailed => write!(f, "CopyFailed"),
            StoreError::WriteFailed => write!(f, "WriteFailed"),
            StoreError::ReadFailed => write!(f, "ReadFailed"),
            StoreError::DeleteFailed => write!(f, "DeleteFailed"),
        }
    }
}

/// What a family gives the cache: copies of its state while the prompt pass
/// stands at a chunk end. Upstream: `Snapshots` (prompt_cache.zig:9) — the
/// required four methods, the optional ones defaulted; the learned-disk and
/// peer hooks ride with `--learn`.
pub trait Snapshots {
    /// The most new storage `save` takes at `at` tokens (checked against
    /// the budget before any copy).
    fn bytes(&self, at: u32) -> u64;
    /// Copy the live state after `at` prompt tokens into new storage;
    /// `owner` is the lane core's stream, opaque the way upstream's
    /// `?*anyopaque` is (the family downcasts it to its own stream type).
    fn save(&mut self, owner: *mut core::ffi::c_void, at: u32) -> Result<Saved, StoreError>;
    /// Make the live state `saved`'s: the next prompt chunk starts at its
    /// position (the serial host's path; a family whose pass restores
    /// itself errors here — upstream's `error.BackendRestores`).
    fn restore(&mut self, owner: *mut core::ffi::c_void, saved: &Saved)
        -> Result<(), StoreError>;
    /// Free a copy.
    fn drop_saved(&mut self, saved: Saved);
    /// The saved state's opaque pointer, for the backend's own restore
    /// (upstream's `Saved` is the thin pointer itself; the port's Box needs
    /// the family's bridge).
    fn saved_ptr(saved: &Saved) -> *mut core::ffi::c_void;
    /// The storage a kept state holds: new storage up to `bytes`, or the
    /// spare storage it took. Upstream: `charged` (None: `bytes`).
    fn charged(&mut self, _saved: &Saved, at: u32) -> u64 {
        self.bytes(at)
    }
    /// Free storage the family keeps for later saves, inside the budget.
    /// Upstream: `spare` (None: none).
    fn spare(&mut self) -> u64 {
        0
    }
    /// Free spare storage until at most `room` bytes stay spare.
    /// Upstream: `trim` (None: none).
    fn trim(&mut self, _room: u64) {}
    /// Whether a save at `at` would take spare storage (no new storage).
    /// Upstream: `reuses` (None: never).
    fn reuses(&mut self, _at: u32) -> bool {
        false
    }
    /// Write `saved` as learned state `key`, in files of the family's own
    /// under `dir` (None: the family keeps nothing on disk — upstream's
    /// null `write` slot). Upstream: `write` (prompt_cache.zig:41).
    fn write(&mut self, _saved: &Saved, _dir: &str, _key: u64) -> Option<Result<(), StoreError>> {
        None
    }
    /// Read learned state `key` back for a pass at `at` (None: the family
    /// reads none). Upstream: `read`.
    fn read(&mut self, _dir: &str, _key: u64, _at: u32) -> Option<Result<Saved, StoreError>> {
        None
    }
    /// Forget learned state `key`'s files. Upstream: `forget` (null: the
    /// index record alone goes).
    fn forget(&mut self, _dir: &str, _key: u64) {}
    /// The checked deletion the learn path's cleanup wants: the files
    /// gone, an error when they cannot be (None: the family checks
    /// nothing — the cleanup refuses). Upstream: `forget_checked`
    /// (prompt_cache.zig:19).
    fn forget_checked(
        &mut self,
        _dir: &str,
        _key: u64,
    ) -> Option<Result<(), StoreError>> {
        None
    }
    /// The bytes deleting learned state `key` would reclaim (None: the
    /// family cannot say — a missing state cannot promise its bytes).
    /// Upstream: `reclaim`.
    fn reclaim(&mut self, _dir: &str, _key: u64) -> Option<Result<u64, StoreError>> {
        None
    }
    /// Whether the `write` slot stands (upstream: `vtable.write != null`),
    /// the learn path's first check.
    fn writes_learned(&self) -> bool {
        false
    }
    /// Whether the `read` slot stands (upstream: `vtable.read != null`),
    /// the recall path's first check.
    fn reads_learned(&self) -> bool {
        false
    }
}

/// The single-rank learned helpers every family shares: the state's own
/// file, `{key}.bin` under the imprint directory. Upstream:
/// `singleReclaim` / `singleForgetChecked` (prompt_cache.zig:11-17).
pub fn single_reclaim(dir: &str, key: u64) -> Result<u64, StoreError> {
    learned_dirs::file_bytes(std::path::Path::new(&state_file(dir, key)))
        .map_err(|_| StoreError::ReadFailed)
}

/// Upstream: `singleForgetChecked` — the acknowledged unlink.
pub fn single_forget_checked(dir: &str, key: u64) -> Result<(), StoreError> {
    learned_dirs::unlink(std::path::Path::new(&state_file(dir, key)))
        .map_err(|_| StoreError::DeleteFailed)
}

fn state_file(dir: &str, key: u64) -> String {
    format!("{dir}/{key:016x}.bin")
}

/// What a family's states depend on beyond their position. Upstream:
/// `Rules` (prompt_cache.zig:59).
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    /// Tokens past the position a state read (Flash Next's MTP head keys
    /// row at-1 with token at: 1).
    pub lookahead: u32,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and
    /// keep only at the request's chunk starts.
    pub planned: bool,
    /// A planned family's own zero-anchored chunk rows, its starts when a
    /// request names none (0: none).
    pub grid: u32,
    /// A mark other than the history's is kept only this far from every
    /// other mark and the resume point.
    pub min_gap: u32,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call
    /// costs more than a later turn's reuse saves.
    pub min_prompt: u32,
    /// Replies are prefilled in background passes kept at their end: a pass
    /// resumed near its history keeps no mark.
    pub warm: bool,
}

impl Default for Rules {
    /// The upstream defaults (prompt_cache.zig:59-72).
    fn default() -> Rules {
        Rules {
            lookahead: 0,
            planned: false,
            grid: 0,
            min_gap: 256,
            min_prompt: 4096,
            warm: false,
        }
    }
}

/// One kept state. Upstream: `Entry` (prompt_cache.zig:77).
pub struct Entry {
    /// The entry's identity: stable across evictions, the pointer upstream's
    /// lookups hand around.
    pub id: u64,
    /// The state's tokens and `lookahead` more.
    pub tokens: Vec<u32>,
    /// Canonical row arithmetic through at, excluding lookahead.
    pub decode_spans: Vec<Span>,
    pub at: u32,
    pub saved: Saved,
    pub bytes: u64,
    /// The length of the prompt that kept it: a later turn's is longer.
    pub born: u32,
    /// The store's clock at its last keep or resume.
    pub used: u64,
    /// The last prompt that kept or resumed it.
    pub last: Vec<u32>,
    /// Kept at a shared system block's cut: other conversations resume it,
    /// so its own never supersedes it.
    pub shared: bool,
}

/// Upstream: `Counts` (prompt_cache.zig:82).
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub hits: u64,
    pub misses: u64,
    pub kept: u64,
    pub evicted: u64,
    pub refused: u64,
    pub failed: u64,
}

/// Where a prompt pass starts and where it stops to keep its state (marks
/// sorted). Upstream: `Plan` (prompt_cache.zig:84).
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub from: u32,
    pub marks: Vec<u32>,
}

/// The entry a backend restores itself (None: the pass starts at 0) and the
/// pass's marks. Upstream: `Lookup` (prompt_cache.zig:86).
pub struct Lookup {
    pub entry: Option<u64>,
    pub marks: Vec<u32>,
}

/// The exact-prompt-reuse store. Upstream: `Store` (prompt_cache.zig:88).
pub struct Store<F: Snapshots> {
    pub(crate) family: F,
    pub(crate) rules: Rules,
    pub(crate) budget: u64,
    pub(crate) held: u64,
    pub(crate) clock: u64,
    pub(crate) entries: Vec<Entry>,
    /// ``--learn``: shared cuts' states on disk, read back by later
    /// sessions and servers (prompt_cache.zig:106). Upstream's store
    /// borrows the host's; the port's owns it, `imprint` the observation.
    pub(crate) imprint: Option<Imprint>,
    pub counts: Counts,
    /// The tokens of recently planned shared cuts, hashed (at most
    /// SHARED_KEYS).
    shared_keys: Vec<u64>,
    /// The last lookup's rewind state (0: none); hosts keep a pass's states
    /// before the next lookup.
    active_rewind_at: u32,
    /// The next entry's id.
    next_id: u64,
}

const SHARED_KEYS: usize = 64;

impl<F: Snapshots> Store<F> {
    /// Upstream: `init` (prompt_cache.zig:100).
    pub fn new(family: F, rules: Rules, budget: u64) -> Store<F> {
        Store {
            family,
            rules,
            budget,
            held: 0,
            clock: 0,
            entries: Vec::new(),
            imprint: None,
            counts: Counts::default(),
            shared_keys: Vec::new(),
            active_rewind_at: 0,
            next_id: 0,
        }
    }

    /// The store learns its shared cuts to `imprint`
    /// (metal.zig:175's `store.imprint = &h.learned`).
    pub fn with_imprint(mut self, imprint: Imprint) -> Store<F> {
        self.imprint = Some(imprint);
        self
    }

    /// The learned states, for observation and the host's own checks.
    pub fn imprint(&self) -> Option<&Imprint> {
        self.imprint.as_ref()
    }

    /// The learned states, mutable (the host's admission wiring).
    pub fn imprint_mut(&mut self) -> Option<&mut Imprint> {
        self.imprint.as_mut()
    }

    /// The family, for the host that saves and restores through the store.
    pub fn family_mut(&mut self) -> &mut F {
        &mut self.family
    }

    /// The rules the store keeps.
    pub fn rules(&self) -> &Rules {
        &self.rules
    }

    /// The kept states, in keep order (upstream's `s.entries.items`).
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The storage the kept states hold (upstream's `s.held`).
    pub fn held(&self) -> u64 {
        self.held
    }

    /// The budget the store keeps inside.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// How many shared cuts the store noted.
    pub fn shared_keys_len(&self) -> usize {
        self.shared_keys.len()
    }

    /// One entry by its id (upstream's pointer).
    pub fn entry(&self, id: u64) -> &Entry {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .expect("the looked-up entry")
    }

    /// The rules, mutable for the tests' flips.
    pub fn rules_mut(&mut self) -> &mut Rules {
        &mut self.rules
    }

    fn remove(&mut self, i: usize) {
        let e = self.entries.remove(i);
        self.held = self.held.saturating_sub(e.bytes);
        self.family.drop_saved(e.saved);
    }

    /// What the budget leaves past the kept states and the spare storage.
    /// Upstream: `room` (prompt_cache.zig:116).
    pub fn room(&mut self) -> u64 {
        self.budget
            .saturating_sub(self.held)
            .saturating_sub(self.family_spare())
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or
    /// a chunk start for planned families. Upstream: `usable`.
    fn usable(&self, at: u32, starts: &[u32]) -> bool {
        at > 0 && (!self.rules.planned || self.start_at(starts, at) == at)
    }

    /// The chunk start at or before `at`: the request's, else the family's
    /// grid (0 when none). Upstream: `startAt`.
    fn start_at(&self, starts: &[u32], at: u32) -> u32 {
        if starts.is_empty() && self.rules.grid > 0 {
            return at - at % self.rules.grid;
        }
        floor_start(starts, at)
    }

    /// The latest kept state this prompt can resume at `at` or under
    /// min_gap before it (None: none). Upstream: `keptNear`.
    fn kept_near(
        &self,
        prompt: &[u32],
        starts: &[u32],
        spans: &[Span],
        at: u32,
    ) -> Option<u32> {
        let mut best: Option<u32> = None;
        for e in &self.entries {
            if e.at > at
                || at.saturating_sub(e.at) >= self.rules.min_gap
                || e.tokens.len() > prompt.len()
                || !self.usable(e.at, starts)
            {
                continue;
            }
            if prompt[..e.tokens.len()] != e.tokens[..]
                || !cache_modes::equal(&e.decode_spans, spans, e.at)
            {
                continue;
            }
            if best.is_none_or(|b| e.at > b) {
                best = Some(e.at);
            }
        }
        best
    }

    /// Find a kept token prefix with matching canonical spans through its
    /// position, leaving at least one row. Upstream: `find`.
    pub fn find(&self, prompt: &[u32], starts: &[u32], spans: &[Span]) -> Option<u64> {
        let mut best: Option<u64> = None;
        for e in self.entries.iter() {
            if e.tokens.len() > prompt.len()
                || e.at >= prompt.len() as u32
                || !self.usable(e.at, starts)
            {
                continue;
            }
            if best.is_some_and(|b| e.at <= self.entry(b).at) {
                continue;
            }
            let n = e.tokens.len();
            if prompt[n - 1] != e.tokens[n - 1] || prompt[..n] != e.tokens[..] {
                continue;
            }
            if !cache_modes::equal(&e.decode_spans, spans, e.at) {
                continue;
            }
            best = Some(e.id);
        }
        best
    }

    /// Restore the longest prefix matching tokens and canonical spans, then
    /// plan prompt-cache marks. Upstream: `begin`.
    pub fn begin(
        &mut self,
        prompt: &[u32],
        history_len: u32,
        shared: &[u32],
        starts: &[u32],
        owner: *mut core::ffi::c_void,
        spans: &[Span],
    ) -> Plan {
        self.begin_rewind(prompt, history_len, 0, shared, starts, owner, spans)
    }

    /// `begin`, also keeping a state before the latest user message
    /// (`rewind_len`, 0: none) that its edits resume. Upstream:
    /// `beginRewind` (prompt_cache.zig:146) — the argument count upstream's
    /// own.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_rewind(
        &mut self,
        prompt: &[u32],
        history_len: u32,
        rewind_len: u32,
        shared: &[u32],
        starts: &[u32],
        owner: *mut core::ffi::c_void,
        spans: &[Span],
    ) -> Plan {
        let l = self.lookup_rewind(prompt, history_len, rewind_len, shared, starts, spans);
        let Some(id) = l.entry else {
            return Plan {
                from: 0,
                marks: l.marks,
            };
        };
        let at = self.entry(id).at;
        let saved = self
            .entries
            .iter()
            .find(|e| e.id == id)
            .map(|e| &e.saved)
            .expect("the looked-up entry");
        let ok = self.family.restore(owner, saved).is_ok();
        let from = if ok { at } else { 0 };
        self.resumed(id, prompt, ok);
        // Restored: the state it resumed may go too (a peer restores before
        // it drops)
        self.reserve(prompt, Some(id), &l.marks, spans);
        Plan {
            from,
            marks: l.marks,
        }
    }

    /// Look up canonical-span state and marks without restoring; the caller
    /// later reports resumed. Upstream: `lookup`.
    pub fn lookup(
        &mut self,
        prompt: &[u32],
        history_len: u32,
        shared: &[u32],
        starts: &[u32],
        spans: &[Span],
    ) -> Lookup {
        self.lookup_rewind(prompt, history_len, 0, shared, starts, spans)
    }

    /// `lookup` plus a rewind state on the starts or grid; a state kept
    /// under min_gap before it serves instead. Upstream: `lookupRewind`
    /// (prompt_cache.zig:196).
    pub fn lookup_rewind(
        &mut self,
        prompt: &[u32],
        history_len: u32,
        rewind_len: u32,
        shared: &[u32],
        starts: &[u32],
        spans: &[Span],
    ) -> Lookup {
        let endpoint = if self.rules.planned {
            self.start_at(starts, history_len)
        } else {
            history_len
        };
        let rewind = if self.rules.planned {
            self.start_at(starts, rewind_len.saturating_sub(self.rules.lookahead))
        } else {
            rewind_len.saturating_sub(self.rules.lookahead)
        };
        let endpoint_bytes = if endpoint > 0
            && (endpoint < prompt.len() as u32
                || (self.rules.warm && endpoint == prompt.len() as u32))
            && endpoint + self.rules.lookahead <= prompt.len() as u32
        {
            self.family.bytes(endpoint)
        } else {
            0
        };
        let rewind_bytes = if rewind > 0
            && rewind < prompt.len() as u32
            && rewind + self.rules.lookahead <= prompt.len() as u32
            && self.usable(rewind, starts)
        {
            self.family.bytes(rewind)
        } else {
            0
        };
        let priority_bytes = if endpoint_bytes <= self.budget {
            endpoint_bytes
        } else {
            0
        };
        self.active_rewind_at = if prompt.len() >= self.rules.min_prompt as usize
            && rewind_bytes > 0
            && rewind != endpoint
            && rewind_bytes <= self.budget.saturating_sub(priority_bytes)
        {
            rewind
        } else {
            0
        };
        if self.active_rewind_at > 0 {
            if let Some(at) = self.kept_near(prompt, starts, spans, self.active_rewind_at) {
                // A state kept under min_gap before it is the rewind state:
                // no new copy
                self.active_rewind_at = at;
            }
        }
        let found = self.find(prompt, starts, spans);
        // A learned state on disk, longer than what memory holds
        // (prompt_cache.zig:427's recall)
        let e = self.recall(prompt, starts, spans, found);
        if e.is_none() {
            self.counts.misses += 1;
        }
        let from = e.map(|id| self.entry(id).at).unwrap_or(0);
        let previous: Vec<u32> = match e {
            Some(id) => self.entry(id).last.clone(),
            None => Vec::new(),
        };
        let marks_ = if self.active_rewind_at == 0 {
            let raw = self.marks(prompt, from, history_len, shared, starts, &previous);
            self.fitting(raw)
        } else {
            let raw = self.marks_rewind(
                prompt,
                from,
                history_len,
                self.active_rewind_at,
                shared,
                starts,
                &previous,
            );
            let fitted = self.fitting(raw);
            self.fitting_rewind(fitted, endpoint, self.active_rewind_at, endpoint_bytes)
        };
        // The shared cuts this pass keeps: their states serve other
        // conversations too
        for &w in shared {
            let at = if self.rules.planned {
                self.start_at(starts, w)
            } else {
                w
            };
            if marks_.contains(&at) {
                let n = (at + self.rules.lookahead) as usize;
                self.note_shared(&prompt[..n]);
            }
        }
        self.reserve(prompt, e, &marks_, spans);
        Lookup {
            entry: e,
            marks: marks_,
        }
    }

    /// Room for every state the pass keeps, made before it. Upstream:
    /// `reserve` (prompt_cache.zig:224).
    fn reserve(
        &mut self,
        prompt: &[u32],
        from: Option<u64>,
        marks: &[u32],
        spans: &[Span],
    ) {
        let mut need: u64 = 0;
        for &at in marks {
            let n = at as usize + self.rules.lookahead as usize;
            let exists = self.entries.iter().any(|e| {
                e.at == at
                    && e.tokens.len() == n
                    && n <= prompt.len()
                    && e.tokens[..] == prompt[..n]
                    && cache_modes::equal(&e.decode_spans, spans, at)
            });
            if !exists {
                need += self.family.bytes(at);
            }
        }
        while self.held + need > self.budget {
            let Some(i) = self.victim_but(from, prompt) else {
                return;
            };
            self.remove(i);
            self.counts.evicted += 1;
        }
    }

    /// Every kept state goes after a weight change. Upstream: `clear`.
    pub fn clear(&mut self) {
        while !self.entries.is_empty() {
            self.remove(self.entries.len() - 1);
        }
        self.shared_keys.clear();
        self.active_rewind_at = 0;
    }

    /// Forget a failed state matching the token prefix and canonical spans
    /// through at. Upstream: `forget`.
    pub fn forget(&mut self, prompt: &[u32], at: u32, spans: &[Span]) {
        for i in 0..self.entries.len() {
            let e = &self.entries[i];
            if e.at == at
                && e.tokens.len() <= prompt.len()
                && e.tokens[..] == prompt[..e.tokens.len()]
                && cache_modes::equal(&e.decode_spans, spans, at)
            {
                self.counts.failed += 1;
                self.remove(i);
                return;
            }
        }
    }

    fn shared_key(tokens: &[u32]) -> u64 {
        // Upstream: Wyhash over the tokens' bytes; the port hashes through
        // the same bytes with the default hasher
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::Hasher;
        for t in tokens {
            h.write_u32(*t);
        }
        h.finish()
    }

    fn note_shared(&mut self, tokens: &[u32]) {
        let k = Self::shared_key(tokens);
        if self.shared_keys.contains(&k) {
            return;
        }
        if self.shared_keys.len() == SHARED_KEYS {
            self.shared_keys.remove(0);
        }
        self.shared_keys.push(k);
    }

    fn family_spare(&mut self) -> u64 {
        self.family.spare()
    }

    /// The marks whose state the budget can hold; the rest are refused now.
    /// Upstream: `fitting` (prompt_cache.zig:252).
    fn fitting(&mut self, marks: Vec<u32>) -> Vec<u32> {
        let mut out = Vec::new();
        for at in marks {
            let bytes = self.family.bytes(at);
            if bytes <= self.budget {
                out.push(at);
            } else {
                self.counts.refused += 1;
            }
        }
        out
    }

    /// Fitted marks that the budget holds together: the history's first,
    /// then the rewind, then the rest in order. Upstream: `fittingRewind`.
    fn fitting_rewind(
        &mut self,
        marks: Vec<u32>,
        endpoint: u32,
        rewind: u32,
        endpoint_bytes: u64,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        let mut used: u64 = 0;
        if endpoint_bytes <= self.budget && endpoint_bytes > 0 {
            used = endpoint_bytes;
            if marks.contains(&endpoint) {
                out.push(endpoint);
            }
        }
        if self.active_rewind_at == rewind && marks.contains(&rewind) {
            out.push(rewind);
        }
        if self.active_rewind_at == rewind {
            used += self.family.bytes(rewind);
        }
        for at in marks {
            if at == endpoint || at == rewind {
                continue;
            }
            let bytes = self.family.bytes(at);
            if bytes <= self.budget.saturating_sub(used) {
                out.push(at);
                used += bytes;
            }
        }
        out.sort_unstable();
        out
    }

    /// A looked-up entry's restore went through (it is now the prompt's),
    /// or failed (dropped: the pass ran from 0). Upstream: `resumed`.
    pub fn resumed(&mut self, id: u64, prompt: &[u32], ok: bool) {
        let Some(i) = self.entries.iter().position(|e| e.id == id) else {
            return;
        };
        if !ok {
            self.counts.failed += 1;
            self.remove(i);
            return;
        }
        self.clock += 1;
        self.entries[i].used = self.clock;
        self.counts.hits += 1;
        self.entries[i].last = prompt.to_vec();
    }

    /// Where a pass from `from` keeps states: the history, then min_gap
    /// apart the stable prefix and shared blocks. Upstream: `marks`.
    fn marks(
        &self,
        prompt: &[u32],
        from: u32,
        history_len: u32,
        shared: &[u32],
        starts: &[u32],
        previous: &[u32],
    ) -> Vec<u32> {
        self.marks_rewind(prompt, from, history_len, 0, shared, starts, previous)
    }

    /// `marks` with the active rewind second: kept however near the other
    /// marks, never at or before the resume point. Upstream: `marksRewind`
    /// (prompt_cache.zig:327) — the argument count upstream's own.
    #[allow(clippy::too_many_arguments)]
    fn marks_rewind(
        &self,
        prompt: &[u32],
        from: u32,
        history_len: u32,
        rewind: u32,
        shared: &[u32],
        starts: &[u32],
        previous: &[u32],
    ) -> Vec<u32> {
        if prompt.len() < self.rules.min_prompt as usize {
            return Vec::new();
        }
        let mut out: Vec<u32> = Vec::new();
        let mut want: Vec<u32> = Vec::new();
        want.push(history_len);
        if rewind > 0 && self.active_rewind_at == rewind {
            want.push(rewind);
        }
        if !previous.is_empty() {
            let stable = previous
                .iter()
                .zip(prompt)
                .position(|(a, b)| a != b)
                .unwrap_or(previous.len().min(prompt.len()));
            let stable = stable as u32;
            if stable > 0 && stable < history_len && stable >= history_len / 2 {
                want.push(stable);
            }
        }
        want.extend_from_slice(shared);
        for (k, &w) in want.iter().enumerate() {
            let at = if self.rules.planned {
                self.start_at(starts, w)
            } else {
                w
            };
            if at <= from
                || at + self.rules.lookahead > prompt.len() as u32
                || at >= prompt.len() as u32
                || !self.usable(at, starts)
            {
                continue;
            }
            let is_rewind = rewind > 0 && k == 1 && at == rewind;
            if !is_rewind
                && at.saturating_sub(from) < self.rules.min_gap
                && (k > 0 || (from > 0 && self.rules.warm))
            {
                continue;
            }
            if k > 0 && !is_rewind {
                // The history's mark always (warm families: away from the
                // resume point); the others away from it and each other
                if out
                    .iter()
                    .any(|&o| o.max(at).saturating_sub(o.min(at)) < self.rules.min_gap)
                {
                    continue;
                }
            }
            if !out.contains(&at) {
                out.push(at);
            }
        }
        out.sort_unstable();
        out
    }

    /// A learned state on disk longer than `have`, read back as a shared
    /// entry; else `have` (none, or reading failed). Upstream: `recall`
    /// (prompt_cache.zig:427).
    fn recall(
        &mut self,
        prompt: &[u32],
        starts: &[u32],
        spans: &[Span],
        have: Option<u64>,
    ) -> Option<u64> {
        if self.imprint.is_none() || !self.family.reads_learned() {
            return have;
        }
        let have_at = have
            .and_then(|id| self.entries.iter().find(|e| e.id == id))
            .map(|e| e.at)
            .unwrap_or(0);
        let best = {
            let im = self.imprint.as_ref().unwrap();
            im.best(prompt, starts, self.rules.planned, have_at)
        };
        let Some(m) = best else {
            return have;
        };
        let (key, at, state_bytes) = (m.key, m.at, m.bytes);
        let tokens = m.tokens.clone();
        // The request decodes rows the learned prompt pass prefilled: a
        // span-carrying request cannot resume it
        if !cache_modes::equal(&[], spans, at) {
            return have;
        }
        while state_bytes <= self.budget
            && self.held + self.family_spare() + state_bytes > self.budget
        {
            let Some(i) = self.victim_but(have, prompt) else {
                return have;
            };
            self.remove(i);
            self.counts.evicted += 1;
        }
        if state_bytes > self.budget {
            return have;
        }
        let dir = self.imprint.as_ref().unwrap().dir.clone();
        let saved = match self.family.read(&dir, key, at) {
            Some(Ok(s)) => s,
            _ => {
                note(format!(
                    "reading the learned {} tokens failed; forgotten, a later pass learns them again",
                    at
                ));
                self.unlearn(key);
                return have;
            }
        };
        self.imprint.as_mut().unwrap().touch(key);
        self.next_id += 1;
        let id = self.next_id;
        self.entries.push(Entry {
            id,
            tokens,
            decode_spans: Vec::new(),
            at,
            saved,
            bytes: state_bytes,
            born: prompt.len() as u32,
            used: self.clock,
            last: prompt.to_vec(),
            shared: true,
        });
        self.held += state_bytes;
        note(format!("read {} learned tokens from disk", at));
        Some(id)
    }

    /// Keep the state at `at` with its span prefix, evicting within the
    /// budget; `starts` are a learned cut's chunks (its disk write's).
    /// Upstream: `keep` (prompt_cache.zig:352).
    pub fn keep(
        &mut self,
        prompt: &[u32],
        at: u32,
        owner: *mut core::ffi::c_void,
        starts: &[u32],
        spans: &[Span],
    ) -> bool {
        let n = at as usize + self.rules.lookahead as usize;
        if at == 0 || n > prompt.len() {
            return false;
        }
        self.clock += 1;
        let shared = self
            .shared_keys
            .contains(&Self::shared_key(&prompt[..n]));
        let same = self.entries.iter().position(|e| {
            e.at == at
                && e.tokens[..] == prompt[..n]
                && cache_modes::equal(&e.decode_spans, spans, at)
        });
        if let Some(i) = same {
            let e = &mut self.entries[i];
            e.used = self.clock; // the same state again: no copy
            e.shared = e.shared || shared;
            if shared {
                let id = e.id;
                self.learn(id, starts);
            }
            return true;
        }
        let decoded = cache_modes::prefix(spans, at);
        let bytes = self.family.bytes(at);
        if bytes > self.budget {
            self.counts.refused += 1;
            return false;
        }
        // Room: spare storage this save can take, else new storage inside
        // the budget
        while !self.entries.is_empty() {
            if self.family.reuses(at) {
                break;
            }
            let free_ = self.family_spare();
            if self.held + free_ + bytes <= self.budget {
                break;
            }
            if free_ > 0 {
                // Spare that cannot take this state goes before any state
                let room = self.budget.saturating_sub(self.held).saturating_sub(bytes);
                self.family.trim(room);
                if self.family_spare() < free_ {
                    continue;
                }
            }
            // Its storage may come back spare and take this state
            let Some(i) = self.victim_but(None, prompt) else {
                break;
            };
            self.remove(i);
            self.counts.evicted += 1;
        }
        if self.entries.is_empty() {
            let room = self.budget.saturating_sub(bytes);
            self.family.trim(room);
        }
        self.next_id += 1;
        let id = self.next_id;
        let saved = match self.family.save(owner, at) {
            Ok(s) => s,
            Err(_) => return self.fail(at),
        };
        let charged = self.family.charged(&saved, at);
        let entry = Entry {
            id,
            tokens: prompt[..n].to_vec(),
            decode_spans: decoded,
            at,
            saved,
            bytes: charged,
            born: prompt.len() as u32,
            used: self.clock,
            last: prompt.to_vec(),
            shared,
        };
        self.held += charged;
        self.counts.kept += 1;
        self.entries.push(entry);
        if shared {
            self.learn(id, starts);
        }
        // Spare storage only inside what the budget leaves
        let room = self.budget.saturating_sub(self.held);
        self.family.trim(room);
        true
    }

    fn fail(&mut self, _at: u32) -> bool {
        self.counts.failed += 1;
        false
    }

    /// The entry to free first, never `skip`: a superseded one, oldest
    /// first; else the oldest, the turn's rewind last. Upstream:
    /// `victimBut` (prompt_cache.zig:483).
    fn victim_but(&self, skip: Option<u64>, prompt: &[u32]) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.shared || Some(e.id) == skip || self.is_active_rewind(e, prompt) {
                continue; // a shared cut is evicted only as the least recently used
            }
            // This prompt extends it
            let now = prompt.len() > e.born as usize
                && prompt.len() > e.tokens.len()
                && prompt[..e.tokens.len()] == e.tokens[..];
            let moved_on = now
                || self.entries.iter().any(|o| {
                    o.id != e.id
                        && o.born > e.born
                        && o.tokens.len() > e.tokens.len()
                        && o.tokens[..e.tokens.len()] == e.tokens[..]
                });
            if moved_on && best.is_none_or(|b| e.used < self.entries[b].used) {
                best = Some(i);
            }
        }
        if best.is_some() {
            return best;
        }
        let mut oldest: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if Some(e.id) != skip
                && oldest.is_none_or(|o| self.before(e, &self.entries[o], prompt))
            {
                oldest = Some(i);
            }
        }
        oldest
    }

    /// Eviction order past the superseded states: least recently used
    /// first, the turn's rewind state last. Upstream: `before`.
    fn before(&self, e: &Entry, o: &Entry, prompt: &[u32]) -> bool {
        let e_rewind = self.is_active_rewind(e, prompt);
        let o_rewind = self.is_active_rewind(o, prompt);
        if e_rewind != o_rewind {
            o_rewind
        } else {
            e.used < o.used
        }
    }

    /// The last lookup's rewind state, while the prompt still extends it.
    /// Upstream: `isActiveRewind`.
    fn is_active_rewind(&self, e: &Entry, prompt: &[u32]) -> bool {
        e.at == self.active_rewind_at
            && e.tokens.len() <= prompt.len()
            && e.tokens[..] == prompt[..e.tokens.len()]
    }
}

/// The chunk start at or before `at` (0 when none). Upstream:
/// `floorStart` (prompt_cache.zig:522).
fn floor_start(starts: &[u32], at: u32) -> u32 {
    let mut best: u32 = 0;
    for &p in starts {
        if p <= at && p > best {
            best = p;
        }
    }
    best
}
/// The store's engine-thread face, the type erasure the lane host holds
/// (upstream's `cache: ?*pc.Store`).
pub trait CacheStore: Send {
    /// The lookup without restoring (the pass restores itself); the caller
    /// later reports `resumed`.
    fn lookup_rewind(
        &mut self,
        prompt: &[u32],
        history_len: u32,
        rewind_len: u32,
        shared: &[u32],
        starts: &[u32],
    ) -> Lookup;
    /// The pass standing at `at`: the state kept (upstream's kept hook).
    fn keep(
        &mut self,
        prompt: &[u32],
        at: u32,
        owner: *mut core::ffi::c_void,
        starts: &[u32],
    ) -> bool;
    /// A restored prefix's report: it went through, or its copy failed.
    fn resumed(&mut self, id: u64, prompt: &[u32], ok: bool);
    /// The saved state's pointer, for the Reuse the backend restores.
    fn saved_ptr(&mut self, id: u64) -> *mut core::ffi::c_void;
    /// The state's position: where the pass restores.
    fn entry_at(&self, id: u64) -> u32;
    /// Every kept state goes (a weight change).
    fn clear(&mut self);
    /// The kept count, for the report line.
    fn counts_kept(&self) -> u64;
}

impl<F: Snapshots + Send> CacheStore for Store<F> {
    fn lookup_rewind(
        &mut self,
        prompt: &[u32],
        history_len: u32,
        rewind_len: u32,
        shared: &[u32],
        starts: &[u32],
    ) -> Lookup {
        Store::lookup_rewind(self, prompt, history_len, rewind_len, shared, starts, &[])
    }
    fn keep(
        &mut self,
        prompt: &[u32],
        at: u32,
        owner: *mut core::ffi::c_void,
        starts: &[u32],
    ) -> bool {
        Store::keep(self, prompt, at, owner, starts, &[])
    }
    fn resumed(&mut self, id: u64, prompt: &[u32], ok: bool) {
        Store::resumed(self, id, prompt, ok)
    }
    fn saved_ptr(&mut self, id: u64) -> *mut core::ffi::c_void {
        let e = self
            .entries
            .iter()
            .find(|e| e.id == id)
            .expect("the looked-up entry");
        F::saved_ptr(&e.saved)
    }
    fn entry_at(&self, id: u64) -> u32 {
        self.entry(id).at
    }
    fn clear(&mut self) {
        Store::clear(self)
    }
    fn counts_kept(&self) -> u64 {
        self.counts.kept
    }
}

#[cfg(test)]
mod tests {
    //! The fake family and the upstream tests (prompt_cache_test.zig:21-
    //! 288, prompt_cache_rewind_test.zig:8-203, the learned cases :290-375),
    //! assertion for assertion — the single-rank learned fault cases in
    //! prompt_learn's tests, the peer pair with the fabric tranche.

    use super::*;
    pub use crate::prompt_cache_fake::{fresh, Fake};

    fn store(rules: Rules, budget: u64) -> (Fake, Store<Fake>) {
        // One shared fake: the test drives it, the store saves through it
        let fake = Fake::default();
        (fake.clone(), Store::new(fake, rules, budget))
    }

    /// Upstream: "a growing conversation resumes each turn where the last
    /// one's history ended, and equals a fresh pass" (prompt_cache_test.zig:21).
    #[test]
    fn a_growing_conversation_resumes_each_turn_where_the_last_ones_history_ended() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 0, ..Default::default() }, 1 << 20);
        let t1: Vec<u32> = vec![1, 2, 3, 4, 5, 9, 9]; // history 5, then a two-token generation prompt
        let p = s.begin(&t1, 5, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 0);
        assert_eq!(p.marks, [5]);
        assert_eq!(fresh(&t1), f.pass(&mut s, &t1, &p));
        let t2: Vec<u32> = vec![1, 2, 3, 4, 5, 9, 7, 7, 6, 6, 9, 9]; // the reply and a tool result, history 10
        let p = s.begin(&t2, 10, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 5);
        assert_eq!(p.marks, [10]);
        assert_eq!(fresh(&t2), f.pass(&mut s, &t2, &p));
        let edited: Vec<u32> = vec![1, 2, 3, 8, 5, 9, 7, 7, 6, 6, 9, 9]; // an earlier turn edited: nothing resumes past it
        let p = s.begin(&edited, 10, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 0);
        assert_eq!(fresh(&edited), f.pass(&mut s, &edited, &p));
        assert_eq!(s.counts.hits, 1);
        assert_eq!(s.counts.misses, 2);
    }

    /// Upstream: "an entry keys its lookahead tokens" (prompt_cache_test.zig:47).
    #[test]
    fn an_entry_keys_its_lookahead_tokens() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 0, ..Default::default() }, 1 << 20);
        let t1: Vec<u32> = vec![1, 2, 3, 4, 5, 9];
        let p = s.begin(&t1, 4, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &t1, &p); // keeps [1 2 3 4] + 5
        assert!(s.find(&[1, 2, 3, 4, 6, 9], &[], &[]).is_none());
        assert_eq!(s.entry(s.find(&[1, 2, 3, 4, 5, 6], &[], &[]).unwrap()).at, 4);
        // The lookahead token must be in the prompt
        assert!(s.find(&[1, 2, 3, 4], &[], &[]).is_none());
    }

    /// Upstream: "planned families resume and keep only at the request's
    /// chunk starts" (prompt_cache_test.zig:62).
    #[test]
    fn planned_families_resume_and_keep_only_at_the_requests_chunk_starts() {
        let (mut f, mut s) = store(Rules { planned: true, min_prompt: 0, ..Default::default() }, 1 << 20);
        let t1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let p = s.begin(&t1, 7, &[], &[4, 6], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [6], "history 7 floored to the start at 6");
        f.pass(&mut s, &t1, &p);
        // 6 is not one of this prompt's starts
        assert!(s.find(&[1, 2, 3, 4, 5, 6, 7, 9], &[4], &[]).is_none());
        let i = s.find(&[1, 2, 3, 4, 5, 6, 7, 9], &[4, 6], &[]).unwrap();
        assert_eq!(s.entry(i).at, 6);
    }

    /// Upstream: "a planned family's own grid gives the starts when a
    /// request names none" (prompt_cache_test.zig:78).
    #[test]
    fn a_planned_familys_own_grid_gives_the_starts_when_a_request_names_none() {
        let (mut f, mut s) = store(
            Rules { lookahead: 1, planned: true, grid: 4, min_prompt: 0, ..Default::default() },
            1 << 20,
        );
        let t1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p1 = s.begin(&t1, 7, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p1.marks, [4], "history 7 floored to the grid's 4");
        f.pass(&mut s, &t1, &p1);
        let t2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14];
        // A request's own starts win over the grid
        assert!(s.find(&t2, &[6], &[]).is_none());
        let p2 = s.begin(&t2, 9, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p2.from, 4);
        assert_eq!(p2.marks, [8]);
        assert_eq!(fresh(&t2), f.pass(&mut s, &t2, &p2));
    }

    /// Upstream: "eviction frees a conversation's superseded state first,
    /// then the oldest; a state past the budget is refused"
    /// (prompt_cache_test.zig:98).
    #[test]
    fn eviction_frees_a_superseded_state_first_then_the_oldest() {
        // Fake bytes: 100 + at
        let (mut f, mut s) = store(Rules { min_prompt: 0, ..Default::default() }, 330);
        let other: Vec<u32> = vec![5, 5, 5, 5];
        let p = s.begin(&other, 3, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &other, &p); // 103 bytes
        let t1: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let p = s.begin(&t1, 4, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &t1, &p); // 104: 207 held
        let t2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let p = s.begin(&t2, 6, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &t2, &p); // 106 more: 313
        assert_eq!(s.entries().len(), 3);
        let t3: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = s.begin(&t3, 8, &[], &[], std::ptr::null_mut(), &[]);
        // 108: t1's state (extended by later turns) goes first
        f.pass(&mut s, &t3, &p);
        assert_eq!(s.entries().len(), 3);
        assert!(s.find(&other, &[], &[]).is_some());
        let i = s.find(&t3, &[], &[]).unwrap();
        assert_eq!(s.entry(i).at, 8);
        let big: Vec<u32> = vec![7; 300];
        let p = s.begin(&big, 299, &[], &[], std::ptr::null_mut(), &[]);
        // 399 > 330: refused, nothing evicted
        f.pass(&mut s, &big, &p);
        assert_eq!(s.counts.refused, 1);
        assert_eq!(s.entries().len(), 3);
        assert!(s.held() <= s.budget());
        assert_eq!(s.entries().len(), f.0.borrow().live);
    }

    /// Upstream: "a failed copy keeps nothing and a failed restore prefills
    /// from the start" (prompt_cache_test.zig:126).
    #[test]
    fn a_failed_copy_keeps_nothing_and_a_failed_restore_prefills_from_the_start() {
        let (mut f, mut s) = store(Rules { min_prompt: 0, ..Default::default() }, 1 << 20);
        f.0.borrow_mut().fail_save = true;
        let t1: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let p = s.begin(&t1, 4, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &t1, &p);
        assert_eq!(s.entries().len(), 0);
        assert_eq!(s.counts.failed, 1);
        f.0.borrow_mut().fail_save = false;
        let p = s.begin(&t1, 4, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &t1, &p);
        assert_eq!(s.entries().len(), 1);
        f.0.borrow_mut().fail_restore = true;
        let t2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
        let p = s.begin(&t2, 6, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 0);
        // The entry that failed is gone
        assert_eq!(s.entries().len(), 0);
        assert_eq!(fresh(&t2), f.pass(&mut s, &t2, &p));
        assert_eq!(s.counts.failed, 2);
    }

    /// Upstream: "marks: the stable prefix with the last prompt and shared
    /// blocks, past the resume point, before the end"
    /// (prompt_cache_test.zig:149).
    #[test]
    fn the_stable_prefix_and_shared_blocks_past_the_resume_point() {
        let (_f, mut s) = store(
            Rules { lookahead: 1, min_gap: 2, min_prompt: 0, ..Default::default() },
            1 << 20,
        );
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let prev: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 0, 0];
        assert_eq!(s.marks(&prompt, 0, 10, &[3, 12, 0], &[], &prev), [3, 7, 10]);
        assert_eq!(s.marks(&prompt, 7, 10, &[3], &[], &prev), [10]);
        // A raw prompt keeps nothing
        assert_eq!(s.marks(&prompt, 0, 0, &[], &[], &[]), Vec::<u32>::new());
        // A block next to the history
        assert_eq!(s.marks(&prompt, 0, 10, &[9], &[], &[]), [10]);
        // Resumed next to the history: kept
        assert_eq!(s.marks(&prompt, 9, 10, &[], &[], &[]), [10]);
        s.rules_mut().warm = true; // replies prefilled in the background: the resumed state serves the next turn
        assert_eq!(s.marks(&prompt, 9, 10, &[], &[], &[]), Vec::<u32>::new());
        // From the start the history's mark stays
        assert_eq!(s.marks(&prompt, 0, 1, &[], &[], &[]), [1]);
    }

    /// Upstream: "prompts under min_prompt keep nothing ... longer ones keep
    /// as before" (prompt_cache_test.zig:169).
    #[test]
    fn prompts_under_min_prompt_keep_nothing() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 8, ..Default::default() }, 1 << 20);
        // 7 tokens, history 5
        let short: Vec<u32> = vec![1, 2, 3, 4, 5, 9, 9];
        let p = s.begin(&short, 5, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, Vec::<u32>::new());
        assert_eq!(fresh(&short), f.pass(&mut s, &short, &p));
        assert_eq!(s.entries().len(), 0);
        // 10 tokens: the next turn keeps its history, resuming nothing
        let long: Vec<u32> = vec![1, 2, 3, 4, 5, 9, 7, 7, 9, 9];
        let q = s.begin(&long, 8, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(q.from, 0);
        assert_eq!(q.marks, [8]);
        assert_eq!(fresh(&long), f.pass(&mut s, &long, &q));
        let i = s.find(&[1, 2, 3, 4, 5, 9, 7, 7, 9, 9, 4], &[], &[]).unwrap();
        assert_eq!(s.entry(i).at, 8);
    }

    /// Upstream: "a shared system cut outlives its own conversation's
    /// turns" (prompt_cache_test.zig:217).
    #[test]
    fn a_shared_system_cut_outlives_its_own_conversations_turns() {
        // Bytes 100 + at
        let (mut f, mut s) = store(
            Rules { min_prompt: 0, min_gap: 1, ..Default::default() },
            600,
        );
        // Cold prompts first
        for c in [vec![9u32; 30], vec![8u32; 30]] {
            let p = s.begin(&c, 29, &[], &[], std::ptr::null_mut(), &[]);
            f.pass(&mut s, &c, &p);
            assert!(s.held() <= s.budget());
        }
        let conv: Vec<u32> = (1..=45u32).collect();
        for (len, history) in [(25u32, 24u32), (35, 34), (45, 44)] {
            let c = &conv[..len as usize];
            let p = s.begin(c, history, &[20], &[], std::ptr::null_mut(), &[]);
            f.pass(&mut s, c, &p);
            assert!(s.held() <= s.budget());
        }
        let c3: Vec<u32> = vec![7; 60];
        let p = s.begin(&c3, 59, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &c3, &p);
        assert!(s.held() <= s.budget());
        let mut other: Vec<u32> = conv[..20].to_vec();
        other.extend((90..95).collect::<Vec<u32>>());
        let plan = s.begin(&other, 24, &[20], &[], std::ptr::null_mut(), &[]);
        // The cut survived the first conversation's turns
        assert_eq!(plan.from, 20);
        assert_eq!(fresh(&other), f.pass(&mut s, &other, &plan));
        assert!(s.held() <= s.budget());
    }

    // ------------------------------------------------------------------
    // The rewind tests (prompt_cache_rewind_test.zig)

    /// Upstream: "editing the latest user resumes the private rewind"
    /// (prompt_cache_rewind_test.zig:8).
    #[test]
    fn editing_the_latest_user_resumes_the_private_rewind() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 0, ..Default::default() }, 211);
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = s.begin_rewind(&first, 8, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [3, 8]);
        assert_eq!(fresh(&first), f.pass(&mut s, &first, &p));
        assert_eq!(s.entries().len(), 2);
        assert!(s.entries().iter().all(|e| !e.shared));
        assert_eq!(s.shared_keys_len(), 0);
        let edited: Vec<u32> = vec![1, 2, 3, 4, 50, 6, 7, 8, 9, 10];
        let p = s.begin_rewind(&edited, 8, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 3);
        assert_eq!(fresh(&edited), f.pass(&mut s, &edited, &p));
        assert_eq!(s.entries().len(), 2);
        assert_eq!(s.entry(s.find(&edited, &[], &[]).unwrap()).at, 8);
        assert_eq!(
            s.entry(s.find(&[1, 2, 3, 4, 51, 6, 7, 8, 9, 10], &[], &[]).unwrap()).at,
            3
        );
    }

    /// Upstream: "one state budget keeps the endpoint before the rewind"
    /// (prompt_cache_rewind_test.zig:33).
    #[test]
    fn one_state_budget_keeps_the_endpoint_before_the_rewind() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 0, ..Default::default() }, 110);
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = s.begin_rewind(&first, 8, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [8]);
        f.pass(&mut s, &first, &p);
        assert_eq!(s.entries()[0].at, 8);
    }

    /// Upstream: "planned rewind floors after lookahead and short user turns
    /// bypass min_gap" (prompt_cache_rewind_test.zig:45).
    #[test]
    fn planned_rewind_floors_after_lookahead() {
        let (mut f, mut s) = store(
            Rules { planned: true, lookahead: 1, min_prompt: 0, ..Default::default() },
            1 << 20,
        );
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let starts = [3u32, 6, 8];
        let p = s.begin_rewind(&first, 9, 5, &[], &starts, std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [3, 8]);
        assert_eq!(fresh(&first), f.pass(&mut s, &first, &p));
        let edited: Vec<u32> = vec![1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11, 12];
        let i = s.find(&edited, &starts, &[]).unwrap();
        assert_eq!(s.entry(i).at, 3);
    }

    /// Upstream: "warm endpoint retains the rewind when two states fit"
    /// (prompt_cache_rewind_test.zig:60).
    #[test]
    fn warm_endpoint_retains_the_rewind_when_two_states_fit() {
        let (mut f, mut s) = store(Rules { warm: true, min_prompt: 0, ..Default::default() }, 214);
        let warm: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = s.begin_rewind(&warm, warm.len() as u32, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [4]);
        f.pass(&mut s, &warm, &p);
        assert!(s.keep(&warm, warm.len() as u32, std::ptr::null_mut(), &[], &[]));
        assert_eq!(s.entries().len(), 2);
        let edited: Vec<u32> = vec![1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11];
        let resumed = s.begin_rewind(&edited, 10, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(resumed.from, 4);
        assert_eq!(fresh(&edited), f.pass(&mut s, &edited, &resumed));
    }

    /// Upstream: "a grid family keeps its rewind on the grid, and an edit
    /// resumes there equal to a fresh pass"
    /// (prompt_cache_rewind_test.zig:75).
    #[test]
    fn a_grid_family_keeps_its_rewind_on_the_grid() {
        let (mut f, mut s) = store(
            Rules { lookahead: 1, planned: true, grid: 4, min_prompt: 0, ..Default::default() },
            1 << 20,
        );
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        // No request starts: the grid's
        let p = s.begin_rewind(&first, 10, 7, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [4, 8]);
        assert_eq!(fresh(&first), f.pass(&mut s, &first, &p));
        let edited: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 50, 9, 10, 11, 12];
        let q = s.begin_rewind(&edited, 10, 7, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(q.from, 4);
        assert_eq!(fresh(&edited), f.pass(&mut s, &edited, &q));
        for e in s.entries() {
            assert_eq!(e.at % 4, 0);
        }
    }

    /// Upstream: "a state kept past the pass's marks evicts the rewind last
    /// instead of failing the keep" (prompt_cache_rewind_test.zig:93).
    #[test]
    fn a_state_kept_past_the_marks_evicts_the_rewind_last() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 0, ..Default::default() }, 215);
        let turn: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = s.begin_rewind(&turn, 8, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [3, 8]);
        f.pass(&mut s, &turn, &p);
        // A reply's state, as the serial host keeps it after decode
        let mut replied = turn.clone();
        replied.extend_from_slice(&[11, 12, 13, 14]);
        f.0.borrow_mut().at = 13;
        assert!(s.keep(&replied, 13, std::ptr::null_mut(), &[], &[]));
        assert!(s.held() <= s.budget());
        assert_eq!(s.entries().len(), 1);
        assert_eq!(s.entries()[0].at, 13);
    }

    /// Upstream: "a first turn's rewind on the system cut stays a shared
    /// cut" (prompt_cache_rewind_test.zig:110).
    #[test]
    fn a_first_turns_rewind_on_the_system_cut_stays_a_shared_cut() {
        let (mut f, mut s) = store(
            Rules { lookahead: 1, planned: true, min_prompt: 0, min_gap: 1, ..Default::default() },
            1 << 20,
        );
        // A system block, the user's text from 4, the generation prompt at 7
        let turn: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // The server's cut before the conversation's own text, and the
        // generation prompt
        let starts = [3u32, 7];
        let p = s.begin_rewind(&turn, 7, 4, &[4], &starts, std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [3, 7]);
        f.pass(&mut s, &turn, &p);
        let cut = s.entries().iter().find(|e| e.at == 3).expect("the cut state");
        assert!(cut.shared);
    }

    /// Upstream: "a history state past the budget is still counted refused
    /// when the rewind is kept" (prompt_cache_rewind_test.zig:126).
    #[test]
    fn a_history_state_past_the_budget_is_counted_refused() {
        let (mut f, mut s) = store(Rules { lookahead: 1, min_prompt: 0, ..Default::default() }, 105);
        // The history at 8 needs 108
        let turn: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = s.begin_rewind(&turn, 8, 4, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.marks, [3]);
        assert_eq!(s.counts.refused, 1);
        assert_eq!(fresh(&turn), f.pass(&mut s, &turn, &p));
    }

    /// Upstream: "a rewind under min_gap past the resume point keeps no
    /// copy: the resumed state serves the edit and goes last"
    /// (prompt_cache_rewind_test.zig:140).
    #[test]
    fn a_rewind_under_min_gap_keeps_no_copy() {
        let (mut f, mut s) = store(
            Rules { warm: true, min_prompt: 0, min_gap: 4, ..Default::default() },
            222,
        );
        let before: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let p0 = s.begin(&before, 6, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &before, &p0); // the state a turn resumes, as a reply's
        let turn: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // the user's text from 8, two tokens past it
        let p = s.begin_rewind(&turn, 11, 8, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 6);
        assert_eq!(p.marks, [11]);
        assert_eq!(fresh(&turn), f.pass(&mut s, &turn, &p));
        let mut replied = turn.clone();
        replied.extend_from_slice(&[13, 14]);
        f.0.borrow_mut().at = 13;
        assert!(s.keep(&replied, 13, std::ptr::null_mut(), &[], &[]));
        assert!(s.held() <= s.budget());
        let edited: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 50, 10, 11, 12];
        let q = s.begin_rewind(&edited, 11, 8, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(q.from, 6);
        assert_eq!(fresh(&edited), f.pass(&mut s, &edited, &q));
    }

    /// Upstream: "the warm pass after a turn keeps that turn's rewind state,
    /// though it resumes later: the edit still resumes there"
    /// (prompt_cache_rewind_test.zig:160).
    #[test]
    fn the_warm_pass_keeps_that_turns_rewind_state() {
        let (mut f, mut s) = store(
            Rules { warm: true, min_prompt: 0, min_gap: 4, ..Default::default() },
            230,
        );
        let before: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let p0 = s.begin(&before, 6, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &before, &p0);
        let turn: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // resumes at 6, the user's text from 8
        let p1 = s.begin_rewind(&turn, 11, 8, &[], &[], std::ptr::null_mut(), &[]);
        f.pass(&mut s, &turn, &p1);
        let mut replied = turn.clone();
        replied.extend_from_slice(&[13, 14]);
        f.0.borrow_mut().at = 13;
        assert!(s.keep(&replied, 13, std::ptr::null_mut(), &[], &[])); // the reply's state
        let mut warm = replied.clone();
        warm.push(15); // the next turn's opening, prefilled in the background from the reply's state
        let w = s.begin_rewind(&warm, warm.len() as u32, 8, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(w.from, 13);
        f.0.borrow_mut().at = 15;
        assert!(s.keep(&warm, 15, std::ptr::null_mut(), &[], &[]));
        assert!(s.held() <= s.budget());
        let edited: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 50, 10, 11, 12];
        let q = s.begin_rewind(&edited, 11, 8, &[], &[], std::ptr::null_mut(), &[]);
        assert_eq!(q.from, 6);
        assert_eq!(fresh(&edited), f.pass(&mut s, &edited, &q));
    }

    /// Upstream's learned end-to-end (prompt_cache_test.zig:290): a
    /// learned harness state outlives its store — a fresh session on a new
    /// one resumes it from disk, equal to a fresh pass.
    #[test]
    fn a_learned_harness_state_outlives_its_store() {
        use crate::prompt_imprint::{key_of, Imprint};
        let root = std::env::temp_dir()
            .join(format!("tf-learn-test-{}", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        let rules = Rules {
            lookahead: 1,
            min_prompt: 0,
            min_gap: 1,
            ..Default::default()
        };
        let harness = [7, 7, 7, 7, 7, 7]; // shared by every session: a state at 5 reads it all (lookahead 1)
        let key = key_of(&harness);
        {
            let im = Imprint::open(&root, 3, 1 << 30).unwrap();
            let mut fake = Fake::default().learned();
            let mut s = Store::new(fake.clone(), rules, 1 << 20).with_imprint(im);
            let mut first = harness.to_vec();
            first.extend_from_slice(&[1, 2, 9]);
            let p = s.begin(&first, 8, &[5], &[], std::ptr::null_mut(), &[]);
            assert_eq!(fresh(&first), fake.pass(&mut s, &first, &p));
            assert!(s.imprint().unwrap().has(key));
        }
        // A new server reads the index back
        let again = Imprint::open(&root, 3, 1 << 30).unwrap();
        let mut fake = Fake::default().learned();
        let mut s = Store::new(fake.clone(), rules, 1 << 20).with_imprint(again);
        let mut second = harness.to_vec();
        second.extend_from_slice(&[3, 4, 9]);
        let p = s.begin(&second, 8, &[5], &[], std::ptr::null_mut(), &[]);
        assert_eq!(p.from, 5, "the fresh store resumed the learned state");
        assert_eq!(fresh(&second), fake.pass(&mut s, &second, &p));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Upstream's learned cap (prompt_cache_test.zig:320): past the cap the
    /// least recently used state is forgotten — and a file that no longer
    /// reads is too, its state unlearned and learned again by a fresh
    /// pass.
    #[test]
    fn past_the_learned_cap_the_least_recently_used_state_is_forgotten() {
        use crate::prompt_imprint::{key_of, Imprint};
        let root = std::env::temp_dir()
            .join(format!("tf-learn-cap-{}", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        let rules = Rules {
            lookahead: 1,
            min_prompt: 0,
            min_gap: 1,
            ..Default::default()
        };
        let one = [7, 7, 7, 7, 7, 7];
        let two = [8, 8, 8, 8, 8, 8];
        // One state plus the conservative index/header allowance fits
        let im = Imprint::open(&root, 3, 440).unwrap();
        let mut fake = Fake::default().learned();
        let mut s = Store::new(fake.clone(), rules, 1 << 20).with_imprint(im);
        let mut p1 = one.to_vec();
        p1.extend_from_slice(&[1, 9]);
        let p = s.begin(&p1, 7, &[5], &[], std::ptr::null_mut(), &[]);
        assert_eq!(fresh(&p1), fake.pass(&mut s, &p1, &p));
        let mut p2 = two.to_vec();
        p2.extend_from_slice(&[1, 9]);
        let p = s.begin(&p2, 7, &[5], &[], std::ptr::null_mut(), &[]);
        assert_eq!(fresh(&p2), fake.pass(&mut s, &p2, &p));
        {
            let im = s.imprint().unwrap();
            assert!(im.has(key_of(&two)));
            assert!(!im.has(key_of(&one)));
            // Its file went too
            assert!(!std::path::Path::new(&format!(
                "{}/{:016x}.bin",
                im.dir,
                key_of(&one)
            ))
            .exists());
        }
        // A file lost behind the store's back
        let dir = s.imprint().unwrap().dir.clone();
        let _ = std::fs::remove_file(format!("{}/{:016x}.bin", dir, key_of(&two)));
        drop(s);
        // The read fails: a fresh pass, and the unreadable state is
        // forgotten — then learned again by that pass
        let im = Imprint::open(&root, 3, 440).unwrap();
        let mut fake = Fake::default().learned();
        let mut t = Store::new(fake.clone(), rules, 1 << 20).with_imprint(im);
        let mut p3 = two.to_vec();
        p3.extend_from_slice(&[2, 9]);
        let plan = t.begin(&p3, 7, &[5], &[], std::ptr::null_mut(), &[]);
        assert_eq!(plan.from, 0, "the unreadable state is not resumed");
        assert!(!t.imprint().unwrap().has(key_of(&two)));
        assert_eq!(fresh(&p3), fake.pass(&mut t, &p3, &plan));
        assert!(
            t.imprint().unwrap().has(key_of(&two)),
            "learned again by that pass"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
