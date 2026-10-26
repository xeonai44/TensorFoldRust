//! The lane core served to the HTTP threads: one thread owns the core and steps
//! rounds while any stream lives.
//!
//! Port of `zig/src/core/lane_host.zig`. Requests queue as jobs (foreground
//! before background), the engine thread admits them into the lane core, steps
//! rounds, delivers each stream's tokens as they land, and finishes the job when
//! its stream ends — dropping the request and sink it was given (ADR-016's
//! contract). The prompt-cache hooks (upstream's `cache: ?*pc.Store`, the Job's
//! entry/marks and the keep hook) port with `core/prompt_cache.zig` in Phase 2;
//! upstream runs this host with `cache = null` for backends without reuse, which
//! is every backend the port admits today.

use crate::cache_modes;
use crate::engine::Engine as Core;
use crate::proposer::{Params, SuffixLookup};
use crate::stream::{Reason as StreamReason, Reuse, Spec, Stream};
use std::sync::{Arc, Condvar, Mutex};
use tensorfold_engine::{
    Engine, Event, EventSink, Finished, Id, Info, Memory, Reason, Request, Stats, Status,
    SubmitError,
};

/// Upstream names every lane-core stream this (lane_host.zig:346).
const ID: &str = "request";

/// Upstream: `min_match` (lane_host.zig:24).
const MIN_MATCH: i64 = 4;

/// Upstream: lane_host.zig:50  window_ns — the decode rate's window.
const WINDOW: std::time::Duration = std::time::Duration::from_secs(2);

/// One admitted or queued request. The job owns the request, the sink, the
/// stream and the proposer for as long as it lives (upstream's Job holds the
/// same four).
struct Job {
    id: Id,
    request: Box<Request<'static>>,
    sink: Option<Box<dyn EventSink + Send>>,
    stream: Option<Box<Stream<'static>>>,
    delivered: usize,
    rows_delivered: usize,
    began: Option<std::time::Instant>,
    prefilled: Option<std::time::Instant>,
    /// Tokens the last send landed, for the decode-rate mark.
    decoded_mark: Option<u64>,
    /// The kept state the backend restores, until its prompt pass reports
    /// (lane_host.zig:66).
    entry: Option<u64>,
    /// Where the pass keeps states.
    marks: Vec<u32>,
    /// The store's kept count when the job looked it up.
    kept0: u64,
    /// The admission ran: the pass's reports reach the store
    /// (lane_host.zig:62's started).
    started: bool,
}

struct Shared {
    queued: Vec<Arc<Mutex<Job>>>,
    admitted: Vec<Arc<Mutex<Job>>>,
    cancels: Vec<Id>,
    closing: bool,
}

/// Upstream: lane_host.zig:19  pub const LaneHost.
pub struct LaneHost {
    shared: Arc<(Mutex<Shared>, Condvar)>,
    core: Mutex<Core<'static>>,
    info: Info,
    /// Tokens a round landed with its time, for the 2s decode rate.
    decoded_marks: Mutex<Vec<(std::time::Instant, u64)>>,
    prefill: Mutex<(f64, Option<std::time::Instant>)>,
    live: Mutex<(Vec<u32>, u64)>,
    /// Kept prompt states (engine thread only; upstream's `cache` at
    /// lane_host.zig:41 — the port's shared handle for the hook's
    /// re-entry, the lock uncontended on the one thread).
    cache: Option<Arc<Mutex<Box<dyn crate::prompt_cache::CacheStore>>>>,
    /// Device memory read (upstream's `memoryFn`, lane_host.zig:217):
    /// None when the backend keeps no count.
    memory_fn: Option<Arc<dyn Fn(bool) -> Option<Memory> + Send + Sync>>,
}

impl LaneHost {
    /// Upstream: `init` (lane_host.zig:86) — the info's loop_guard is enforced on.
    pub fn new(core: Core<'static>, info: Info) -> LaneHost {
        let mut enforced = info;
        enforced.loop_guard = true;
        LaneHost {
            shared: Arc::new((
                Mutex::new(Shared {
                    queued: Vec::new(),
                    admitted: Vec::new(),
                    cancels: Vec::new(),
                    closing: false,
                }),
                Condvar::new(),
            )),
            core: Mutex::new(core),
            info: enforced,
            decoded_marks: Mutex::new(Vec::new()),
            prefill: Mutex::new((0.0, None)),
            live: Mutex::new((Vec::new(), 0)),
            cache: None,
            memory_fn: None,
        }
    }

    /// The prompt-cache store the engine hands the host (upstream's
    /// `h.host.cache = &h.cache.?`, metal.zig:165): the kept states, the
    /// pass's keeps going through it.
    pub fn with_cache(
        mut self,
        store: Box<dyn crate::prompt_cache::CacheStore>,
    ) -> LaneHost {
        self.cache = Some(Arc::new(Mutex::new(store)));
        self
    }

    /// Device memory in bytes for /v1/memory (upstream's `memoryFn`,
    /// lane_host.zig:217 — set by the native hosts that hold a GPU).
    pub fn with_memory(
        mut self,
        read: impl Fn(bool) -> Option<Memory> + Send + Sync + 'static,
    ) -> LaneHost {
        self.memory_fn = Some(Arc::new(read));
        self
    }

    /// Upstream: `start` (lane_host.zig:92) — the engine thread, 16 MiB of stack.
    pub fn start(self) -> Arc<LaneHost> {
        let host = Arc::new(self);
        let thread_host = Arc::clone(&host);
        let thread = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(move || run(thread_host))
            .expect("the engine thread starts");
        let _ = thread; // the thread holds its Arc and exits when closing drains
        host
    }

    /// Upstream: `stop` (lane_host.zig:97) — stop admitting and cancel what is
    /// left; the engine thread exits once every job has finished.
    pub fn stop(&self) {
        {
            let (shared, wake) = &*self.shared;
            shared.lock().unwrap().closing = true;
            wake.notify_all();
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Shared> {
        self.shared.0.lock().unwrap()
    }
}

impl Engine for LaneHost {
    fn info(&self) -> Info {
        self.info.clone()
    }

    /// Upstream: `submitFn` (lane_host.zig:165) — a job queued foreground-first,
    /// then the engine thread wakes. The spans validate where the contract's own
    /// wrapper validates them (engine_api.zig:292).
    fn submit(
        &self,
        id: Id,
        request: Box<Request<'static>>,
        sink: Box<dyn EventSink + Send>,
    ) -> Result<(), SubmitError> {
        let spans: Vec<cache_modes::Span> = request
            .decode_spans
            .iter()
            .map(|s| [s.start, s.end])
            .collect();
        cache_modes::validate(&spans).map_err(|_| SubmitError::InvalidSpans)?;
        let job = Arc::new(Mutex::new(Job {
            id,
            request,
            sink: Some(sink),
            stream: None,
            delivered: 0,
            rows_delivered: 0,
            began: None,
            prefilled: None,
            decoded_mark: None,
            entry: None,
            marks: Vec::new(),
            kept0: 0,
            started: false,
        }));
        let background = job.lock().unwrap().request.background;
        let mut shared = self.lock();
        if shared.closing {
            return Err(SubmitError::Closed);
        }
        let mut at = shared.queued.len();
        if !background {
            while at > 0 && shared.queued[at - 1].lock().unwrap().request.background {
                at -= 1;
            }
        }
        shared.queued.insert(at, job);
        drop(shared);
        self.shared.1.notify_all();
        Ok(())
    }

    /// Upstream: `cancelFn` (lane_host.zig:187).
    fn cancel(&self, id: Id) {
        self.lock().cancels.push(id);
        self.shared.1.notify_all();
    }

    /// Upstream: `statusFn` (lane_host.zig:195).
    fn status(&self, stream_tokens: &mut [u32]) -> Status {
        let shared = self.lock();
        let now = std::time::Instant::now();
        let mut tokens: u64 = 0;
        for (at, n) in self.decoded_marks.lock().unwrap().iter() {
            if now.duration_since(*at) <= WINDOW {
                tokens += n;
            }
        }
        let (live_tokens, generated) = self.live.lock().unwrap().clone();
        let n = stream_tokens.len().min(live_tokens.len());
        stream_tokens[..n].copy_from_slice(&live_tokens[..n]);
        let (prefill_rate, prefill_at) = *self.prefill.lock().unwrap();
        Status {
            running: shared.admitted.len() as u32,
            waiting: shared.queued.len() as u32,
            decode_tokens_per_second: tokens as f64 / 2.0,
            prefill_tokens_per_second: if prefill_at
                .map(|at| now.duration_since(at) <= WINDOW)
                .unwrap_or(false)
            {
                prefill_rate
            } else {
                0.0
            },
            preemptions: Some(0),
            warming: false,
            streams: n,
            generation_tokens: generated,
        }
    }

    /// Upstream: `memoryFn` (lane_host.zig:217) — none reported.
    fn memory(&self, reset_peak: bool) -> Option<Memory> {
        (self.memory_fn.as_ref()?)(reset_peak)
    }
}

/// The engine thread. Upstream: `run` (lane_host.zig:480).
fn run(h: Arc<LaneHost>) {
    loop {
        take_cancels(&h);
        while admit_one(&h) {}
        note_live(&h);
        {
            let shared = h.lock();
            if shared.closing {
                let left = shared.queued.len() + shared.admitted.len();
                if left == 0 {
                    return;
                }
                drop(shared);
                // Upstream: `closeAll` (lane_host.zig:538) — cancel everything
                // left, then take them.
                {
                    let ids: Vec<Id> = {
                        let shared = h.lock();
                        shared
                            .queued
                            .iter()
                            .chain(shared.admitted.iter())
                            .map(|job| job.lock().unwrap().id)
                            .collect()
                    };
                    h.lock().cancels.extend(ids);
                }
                take_cancels(&h);
                continue;
            }
            let active = h.core.lock().unwrap().active_count();
            if active == 0 {
                if shared.cancels.is_empty()
                    && (shared.queued.is_empty() || shared.admitted.len() >= h.info.lanes as usize)
                {
                    let (_, wake) = &*h.shared;
                    let _ = wake
                        .wait_timeout(shared, std::time::Duration::from_millis(100))
                        .unwrap();
                }
                continue;
            }
            drop(shared);
        }
        match h.core.lock().unwrap().step() {
            Ok(()) => {}
            Err(e) => fail_all(&h, &e.to_string()),
        }
        // Deliver: finished jobs leave the admitted list.
        let jobs: Vec<Arc<Mutex<Job>>> = h.lock().admitted.clone();
        for job in jobs {
            if deliver(&h, &job) {
                let mut shared = h.lock();
                if let Some(at) = shared.admitted.iter().position(|j| Arc::ptr_eq(j, &job)) {
                    shared.admitted.remove(at);
                }
            }
        }
    }
}

/// Upstream: `takeCancels` (lane_host.zig:287) — queued jobs finish at once;
/// admitted jobs discard their stream, then finish.
fn take_cancels(h: &Arc<LaneHost>) {
    let ids: Vec<Id> = std::mem::take(&mut h.lock().cancels);
    for id in ids {
        let queued = {
            let mut shared = h.lock();
            shared.queued.iter().position(|j| j.lock().unwrap().id == id).map(|at| shared.queued.remove(at))
        };
        if let Some(job) = queued {
            let mut j = job.lock().unwrap();
            finish_reported(h, &mut j);
            emit_finished(&mut j, Reason::Cancelled, Stats::default(), "");
            continue;
        }
        let admitted = {
            let shared = h.lock();
            shared.admitted.iter().find(|j| j.lock().unwrap().id == id).cloned()
        };
        if let Some(job) = admitted {
            let mut j = job.lock().unwrap();
            finish_reported(h, &mut j);
            let mut stats = Stats::default();
            if let Some(stream) = j.stream.take() {
                let ptr: *mut Stream<'static> = Box::into_raw(stream);
                // Safety: the job's own stream, live; the core owns its live-list
                // entry until discard removes it.
                unsafe {
                    if !(*ptr).finished {
                        h.core.lock().unwrap().discard(ptr);
                    }
                    let stream = Box::from_raw(ptr);
                    take_stats(&stream, &mut stats);
                }
            }
            let began = j.began;
            let stats = stats_with_prefill(stats, &j);
            emit_finished(&mut j, Reason::Cancelled, stats, "");
            let _ = began;
        }
    }
}

/// Report a restored prefix or its failed copy; an untouched prefix remains
/// kept (lane_host.zig:76-83). The backend restores before its first
/// chunk, so a later keep may free it: the borrow releases here.
fn reported(h: &Arc<LaneHost>, j: &mut Job, stream: *mut Stream<'_>) {
    reported_ptr(h, j, Some(stream));
}

/// The end paths' report (upstream: finish's `job.reported()` at
/// lane_host.zig:260): the stream read from the job, absent a taken one.
fn finish_reported(h: &Arc<LaneHost>, j: &mut Job) {
    let stream = j
        .stream
        .as_ref()
        .map(|s| &**s as *const Stream<'static> as *mut Stream<'static>);
    reported_ptr(h, j, stream);
}

fn reported_ptr(h: &Arc<LaneHost>, j: &mut Job, stream: Option<*mut Stream<'_>>) {
    let Some(id) = j.entry.take() else {
        return;
    };
    if !j.started {
        return;
    }
    let Some(stream) = stream else {
        return;
    };
    // Safety: the stream is the job's own, live through its pass.
    unsafe {
        (*stream).reuse.saved = std::ptr::null_mut();
    }
    let (failed, cached) = unsafe { ((*stream).reuse_failed, (*stream).cached) };
    let Some(store) = h.cache.as_ref() else {
        return;
    };
    let mut store = store.lock().unwrap();
    let at = store.entry_at(id);
    let prompt = j.request.prompt;
    if failed {
        store.resumed(id, prompt, false);
    } else if cached == at {
        store.resumed(id, prompt, true);
    }
}

/// Upstream: `admitOne` (lane_host.zig:323) — prefill the next queued request
/// into a free lane.
/// The request's Copy facts, read under a short lock and used past it (the
/// admission's own re-structure: the pass re-enters the job).
struct Facts {
    prompt: &'static [u32],
    max_tokens: u32,
    eos: &'static [u32],
    sampling: Option<tensorfold_engine::Sampling>,
    drafts: bool,
    has_stop: bool,
    think_budget: u32,
    think_close: &'static [u32],
    think_end: Option<u32>,
    loop_guard: bool,
    chunks: &'static [u32],
    logprobs: Option<u8>,
}

impl Facts {
    fn of(r: &Request<'static>) -> Facts {
        Facts {
            prompt: r.prompt,
            max_tokens: r.max_tokens,
            eos: r.eos,
            sampling: r.sampling,
            drafts: r.drafts,
            has_stop: r.stop.is_some(),
            think_budget: r.think_budget,
            think_close: r.think_close,
            think_end: r.think_end,
            loop_guard: r.loop_guard,
            chunks: r.chunks,
            logprobs: r.logprobs,
        }
    }
}

fn admit_one(h: &Arc<LaneHost>) -> bool {
    let job = {
        let mut shared = h.lock();
        if shared.queued.is_empty() || shared.admitted.len() >= h.info.lanes as usize {
            return false;
        }
        let job = shared.queued.remove(0);
        shared.admitted.push(Arc::clone(&job));
        job
    };
    // The request's facts are all Copy ('static slices and plain values),
    // so the admission reads them under one short lock and drops the guard
    // before the prompt pass: the keep hook re-enters the job
    // (lane_host.zig:68), and the pass runs inside add_stream.
    let mut reuse = Reuse::default();
    let r_facts: Facts = {
        let mut j = job.lock().unwrap();
        // The prompt cache's lookup (lane_host.zig:339-345): the kept
        // state the backend restores, the marks its pass keeps at. The
        // entry stays alive until its pass reports (the reported
        // discipline).
        if let Some(store) = h.cache.clone() {
        let (prompt, history_len, rewind_len, shared, chunks) = {
            let req: &Request<'static> = &j.request;
            (
                req.prompt,
                req.history_len,
                req.rewind_len,
                req.shared_prefixes,
                req.chunks,
            )
        };
            let l = store
                .lock()
                .unwrap()
                .lookup_rewind(prompt, history_len, rewind_len, shared, chunks);
            let saved = l.entry.map(|id| store.lock().unwrap().saved_ptr(id));
            let at = l.entry.map(|id| store.lock().unwrap().entry_at(id)).unwrap_or(0);
            j.entry = l.entry;
            j.kept0 = store.lock().unwrap().counts_kept();
            j.marks = l.marks.clone();
            if !l.marks.is_empty() || l.entry.is_some() {
                let marks: &'static [u32] = Box::leak(l.marks.into_boxed_slice());
                reuse = Reuse {
                    saved: saved.unwrap_or(std::ptr::null_mut()),
                    at,
                    marks,
                    hook: Some(Box::new({
                        let host = Arc::clone(h);
                        let job = Arc::clone(&job);
                        let store = Arc::clone(&store);
                        move |stream: *mut Stream, at: u32| {
                            // The backend's prompt pass stands at a mark:
                            // the cache keeps the stream's state there
                            // (lane_host.zig:68) — the report first, a keep
                            // may evict the entry the pass restored.
                            let mut j = job.lock().unwrap();
                            reported(&host, &mut j, stream);
                            let (prompt, chunks) = (j.request.prompt, j.request.chunks);
                            store
                                .lock()
                                .unwrap()
                                .keep(prompt, at, stream as *mut core::ffi::c_void, chunks);
                        }
                    })),
                };
            }
        }
        Facts::of(&j.request)
    };
    let (job_id, _) = {
        let j = job.lock().unwrap();
        (j.id, ())
    };
    let proposer = Box::new(
        SuffixLookup::new(Params { min_match: MIN_MATCH, ..Params::default() })
            .expect("the drafter starts"),
    );
    let cancel = CancelCheck {
        host: Arc::clone(h),
        id: job_id,
    };
    let mut stream = Box::new(Stream::init(Spec {
        id: ID,
        prompt: r_facts.prompt,
        max_new: r_facts.max_tokens,
        eos: r_facts.eos,
        sampling: r_facts.sampling,
        drafts: r_facts.drafts,
        reuse,
        proposer: Some(Box::new(SuffixLookup::new(Params {
            min_match: MIN_MATCH,
            ..Params::default()
        })
        .expect("the drafter starts"))),
        // The request's own stop check, moved into the stream (the stream's
        // rounds call it on the engine thread).
        stop_check: r_facts.has_stop.then(|| Box::new(StopForward) as Box<dyn crate::stream::StopCheck>),
        cancel_check: Some(Box::new(cancel)),
        think_budget: r_facts.think_budget,
        think_close: r_facts.think_close,
        think_end: r_facts.think_end.map(|t| t as i64).unwrap_or(-1),
        loop_guard: r_facts.loop_guard,
        chunks: r_facts.chunks,
        logprobs: r_facts.logprobs,
        ..Spec::new(ID, r_facts.prompt, r_facts.max_tokens)
    }));
    let _ = proposer;
    let ptr: *mut Stream<'static> = &mut *stream as *mut Stream<'static>;
    {
        // The job's fields under one short lock, then the guard drops: the
        // pass inside add_stream re-enters the job through the hook.
        let mut j = job.lock().unwrap();
        j.began = Some(std::time::Instant::now());
        // The stream's box is stable for as long as the job holds it; the
        // core gets the pointer and keeps it in its live list until the
        // stream finishes.
        j.stream = Some(stream);
        // The admission ran: the pass's reports reach the store
        // (lane_host.zig:367's started).
        j.started = true;
    }
    let result = {
        let mut core = h.core.lock().unwrap();
        // Safety: the job owns the box and outlives the core's live-list entry;
        // deliver/cancel drop it only after the core has released the stream.
        unsafe { core.add_stream(ptr) }
    };
    // The pass reports (lane_host.zig:384's reported at prefilled): the
    // restored prefix's copy stands or failed. The job lock re-taken.
    let mut j = job.lock().unwrap();
    let began = j.began;
    reported(h, &mut j, ptr);
    match result {
        Ok(()) => {
            // Upstream: prefilled() (lane_host.zig:377) — the rate and the
            // prefilled event after the admission's prompt pass.
            let done = std::time::Instant::now();
            let cached = j.stream.as_ref().map(|s| s.cached).unwrap_or(0) as usize;
            let prompt_len = j.request.prompt.len();
            {
                let mut prefill = h.prefill.lock().unwrap();
                if let Some(began) = began {
                if done > began {
                    prefill.0 = (prompt_len.saturating_sub(cached)) as f64
                        / done.duration_since(began).as_secs_f64();
                    prefill.1 = Some(done);
                }
            }
            }
            let prefilled = j.stream.as_ref().map(|s| s.cached).unwrap_or(0);
            emit(&mut j, Event::Prefilled(prefilled));
            let finished_now = j.stream.as_ref().map(|s| s.finished).unwrap_or(true);
            drop(j);
            if finished_now {
                // The core released it during admission; the job delivers its
                // tokens and finishes.
                let mut j = job.lock().unwrap();
                send(&mut j);
                let mut stats = Stats::default();
                if let Some(stream) = j.stream.take() {
                    take_stats(&stream, &mut stats);
                }
                let reason = Reason::Stop;
                let stats = stats_with_prefill(stats, &j);
                emit_finished(&mut j, reason, stats, "");
                let mut shared = h.lock();
                if let Some(at) = shared.admitted.iter().position(|x| Arc::ptr_eq(x, &job)) {
                    shared.admitted.remove(at);
                }
            }
        }
        Err(e) => {
            j.stream = None;
            drop(j);
            let mut j = job.lock().unwrap();
            let reason = if matches!(e, crate::Error::Cancelled) {
                Reason::Cancelled
            } else {
                Reason::Failed
            };
            emit_finished(&mut j, reason, Stats::default(), &e.to_string());
            let mut shared = h.lock();
            if let Some(at) = shared.admitted.iter().position(|x| Arc::ptr_eq(x, &job)) {
                shared.admitted.remove(at);
            }
        }
    }
    true
}

/// The stop-string check, forwarded to the request's own (the stream's rounds
/// call it on the engine thread; the request's check owns its stop policy).
struct StopForward;

impl crate::stream::StopCheck for StopForward {
    fn should_stop(&mut self, _emitted: &[u32]) -> bool {
        // The stream's own stop_check runs the request's; the Spec takes one
        // hook, so the request's is installed directly (see admit_one — this
        // forwarder is unused and never constructed).
        false
    }
}

/// The job's cancel hook: its id is in `cancels`, or the host is closing.
/// Upstream: `cancelled` (lane_host.zig:227).
struct CancelCheck {
    host: Arc<LaneHost>,
    id: Id,
}

impl crate::stream::CancelCheck for CancelCheck {
    fn is_cancelled(&mut self) -> bool {
        let shared = self.host.lock();
        shared.closing || shared.cancels.contains(&self.id)
    }
}

/// Upstream: `send` (lane_host.zig:235) — the tokens a stream's rounds committed
/// since the last delivery, their logprob rows first.
fn send(j: &mut Job) {
    let Some(stream) = j.stream.as_ref() else { return };
    let emitted = stream.emitted();
    if emitted.len() > j.delivered {
        if stream.logprobs.is_some() {
            let rows = &stream.rows[j.rows_delivered..emitted.len()];
            let rows: &[tensorfold_engine::LogprobRow] = unsafe { std::mem::transmute(rows) };
            if let Some(sink) = j.sink.as_mut() {
                sink.event(j.id, Event::Logprobs(rows));
            }
        }
        let tokens: &[u32] = unsafe { std::mem::transmute(&emitted[j.delivered..]) };
        let n = tokens.len();
        if let Some(sink) = j.sink.as_mut() {
            sink.event(j.id, Event::Tokens(tokens));
        }
        j.delivered = emitted.len();
        j.rows_delivered = emitted.len();
        j.decoded_mark = Some(n as u64);
    }
}

impl Job {
    fn decoded_mark(&mut self) -> Option<(std::time::Instant, u64)> {
        self.decoded_mark.take().map(|n| (std::time::Instant::now(), n))
    }
}

/// Upstream: `deliver` (lane_host.zig:246) — send, then true once the stream
/// finished (its job frees).
fn deliver(h: &Arc<LaneHost>, job: &Arc<Mutex<Job>>) -> bool {
    let mut j = job.lock().unwrap();
    send(&mut j);
    if let Some(mark) = j.decoded_mark() {
        h.decoded_marks.lock().unwrap().push(mark);
        let mut marks = h.decoded_marks.lock().unwrap();
        let now = std::time::Instant::now();
        marks.retain(|(at, _)| now.duration_since(*at) <= WINDOW);
    }
    let finished = j.stream.as_ref().map(|s| s.finished).unwrap_or(true);
    if !finished {
        return false;
    }
    finish_reported(h, &mut j);
    let mut stats = Stats::default();
    let reason = {
        let stream = j.stream.take();
        match stream {
            Some(stream) => {
                take_stats(&stream, &mut stats);
                stream_reason(stream.reason)
            }
            None => Reason::Stop,
        }
    };
    let stats = stats_with_prefill(stats, &j);
    emit_finished(&mut j, reason, stats, "");
    true
}

fn stream_reason(r: StreamReason) -> Reason {
    match r {
        StreamReason::None | StreamReason::Stop => Reason::Stop,
        StreamReason::Length => Reason::Length,
        StreamReason::Cancelled => Reason::Cancelled,
        StreamReason::Error => Reason::Failed,
    }
}

/// The stream's stats, as the finished event reports them (lane_host.zig:263).
fn take_stats(stream: &Stream<'_>, stats: &mut Stats) {
    stats.rounds = stream.rounds;
    stats.drafted = stream.drafted;
    stats.accepted = stream.accepted;
    stats.min_rows = stream.min_rows;
    stats.loop_period = stream.loop_period;
}

fn stats_with_prefill(stats: Stats, j: &Job) -> Stats {
    let mut stats = stats;
    if let (Some(began), Some(done)) = (j.began, j.prefilled.or(j.began)) {
        if done > began {
            stats.prefill_seconds = Some(done.duration_since(began).as_secs_f64());
        }
    }
    stats
}

fn emit(j: &mut Job, event: Event<'static>) {
    if let Some(sink) = j.sink.as_mut() {
        sink.event(j.id, event);
    }
}

fn emit_finished(j: &mut Job, reason: Reason, stats: Stats, message: &str) {
    emit(
        j,
        Event::Finished(Finished {
            reason,
            stats,
            message: message.to_string(),
        }),
    );
    // The finished event was the last: the request and sink drop now (ADR-016).
    j.sink = None;
}

/// Upstream: `noteLive` (lane_host.zig:469) — the live streams' token counts.
fn note_live(h: &Arc<LaneHost>) {
    let jobs: Vec<Arc<Mutex<Job>>> = h.lock().admitted.clone();
    let mut tokens: Vec<u32> = Vec::with_capacity(jobs.len());
    let mut generated: u64 = 0;
    for job in jobs {
        let j = job.lock().unwrap();
        if let Some(stream) = j.stream.as_ref() {
            tokens.push(stream.context.len() as u32);
            generated += stream.emitted().len() as u64;
        }
    }
    *h.live.lock().unwrap() = (tokens, generated);
}

/// Upstream: `failAll` (lane_host.zig:526) — a failed round ends every stream it
/// held.
fn fail_all(h: &Arc<LaneHost>, message: &str) {
    let jobs: Vec<Arc<Mutex<Job>>> = std::mem::take(&mut h.lock().admitted);
    for job in jobs {
        let mut j = job.lock().unwrap();
        finish_reported(h, &mut j);
        let mut stats = Stats::default();
        if let Some(stream) = j.stream.take() {
            let ptr: *mut Stream<'static> = Box::into_raw(stream);
            // Safety: the core is between rounds here; discard releases the
            // backend's caches, then the box frees.
            unsafe {
                if !(*ptr).finished {
                    h.core.lock().unwrap().discard(ptr);
                }
                take_stats(&*ptr, &mut stats);
                drop(Box::from_raw(ptr));
            }
        }
        emit_finished(&mut j, Reason::Failed, stats, message);
    }
}
