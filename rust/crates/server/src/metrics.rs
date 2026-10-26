//! Prometheus text for GET /metrics, family for family as the Python server
//! writes it.
//!
//! Port of `zig/src/server/metrics.zig` (359 lines) at the fidelity the port's
//! engine admits: the request counters, the arrival/decode/prefill histograms
//! and the engine gauges, with the vLLM aliases upstream carries. The process
//! footprint gauges (macOS's physical footprint via task_info) and the engine's
//! device-memory counters (CUDA) arrive with their backends.

use tensorfold_engine::{Engine, Status};
use std::fmt::Write as _;
use std::sync::Mutex;

/// Upstream: metrics.zig:7  pub const prefix = "tensorfold:".
pub const PREFIX: &str = "tensorfold:";

/// Upstream: metrics.zig:9  pub const buckets.
pub const BUCKETS: [f64; 13] = [
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Upstream: metrics.zig:10  pub const tpot_buckets.
pub const TPOT_BUCKETS: [f64; 17] = [
    0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1,
    0.15, 0.2, 0.3, 0.5, 1.0,
];

/// One histogram over fixed edges. Upstream: metrics.zig:12 HistogramType.
#[derive(Default)]
pub struct Histogram {
    pub edges: &'static [f64],
    pub counts: Vec<u64>,
    pub total: f64,
    pub n: u64,
}

impl Histogram {
    pub fn new(edges: &'static [f64]) -> Histogram {
        Histogram {
            edges,
            counts: vec![0; edges.len() + 1],
            total: 0.0,
            n: 0,
        }
    }

    /// Upstream: `observe` — the +Inf bucket is the last count.
    pub fn observe(&mut self, raw: f64) {
        let value = raw.max(0.0);
        self.n += 1;
        self.total += value;
        for (i, edge) in self.edges.iter().enumerate() {
            if value <= *edge {
                self.counts[i] += 1;
                return;
            }
        }
        self.counts[self.edges.len()] += 1;
    }
}

/// Counters of finished requests; gauges are read from the engine at scrape
/// time. Upstream: metrics.zig:31  pub const Metrics.
#[derive(Default)]
pub struct Metrics {
    state: Mutex<MetricsInner>,
}

#[derive(Default)]
struct MetricsInner {
    prompt: u64,
    cached: u64,
    generation: u64,
    drafted: u64,
    accepted: u64,
    rounds: u64,
    disconnects: u64,
    latency: Option<Histogram>,
    ttft: Option<Histogram>,
    decode: Option<Histogram>,
    prefill: Option<Histogram>,
    tpot: Option<Histogram>,
    requests: Vec<(String, u16, u64)>,
}

impl Metrics {
    pub fn new() -> Metrics {
        Metrics {
            state: Mutex::new(MetricsInner {
                latency: Some(Histogram::new(&BUCKETS)),
                ttft: Some(Histogram::new(&BUCKETS)),
                decode: Some(Histogram::new(&BUCKETS)),
                prefill: Some(Histogram::new(&BUCKETS)),
                tpot: Some(Histogram::new(&TPOT_BUCKETS)),
                ..MetricsInner::default()
            }),
        }
    }

    /// Upstream: `note` (metrics.zig:51).
    #[allow(clippy::too_many_arguments)]
    pub fn note(
        &self,
        prompt: usize,
        cached: usize,
        generation: usize,
        drafted: u64,
        accepted: u64,
        rounds: u64,
        latency: f64,
        ttft: Option<f64>,
        decode: Option<f64>,
        prefill: Option<f64>,
        tpot: Option<f64>,
    ) {
        let mut m = self.state.lock().unwrap();
        m.prompt += prompt as u64;
        m.cached += cached as u64;
        m.generation += generation as u64;
        m.drafted += drafted;
        m.accepted += accepted;
        m.rounds += rounds;
        if let Some(h) = m.latency.as_mut() {
            h.observe(latency);
        }
        if let (Some(h), Some(t)) = (m.ttft.as_mut(), ttft) {
            h.observe(t);
        }
        if let (Some(h), Some(d)) = (m.decode.as_mut(), decode) {
            h.observe(d);
        }
        if let (Some(h), Some(p)) = (m.prefill.as_mut(), prefill) {
            h.observe(p);
        }
        if let (Some(h), Some(t)) = (m.tpot.as_mut(), tpot) {
            h.observe(t);
        }
    }

    /// One HTTP reply under its key label, counted once per request.
    /// Upstream: `httpRequest` (metrics.zig:66).
    pub fn http_request(&self, key: &str, status: u16) {
        let mut m = self.state.lock().unwrap();
        for r in m.requests.iter_mut() {
            if r.1 == status && r.0 == key {
                r.2 += 1;
                return;
            }
        }
        m.requests.push((key.to_string(), status, 1));
    }

    /// Upstream: `disconnected` (metrics.zig:75).
    pub fn disconnected(&self) {
        self.state.lock().unwrap().disconnects += 1;
    }

    /// The scrape body, ending in a newline. Upstream: `render`
    /// The counters the /stats route reads (upstream: the mutex-guarded
    /// reads at status_routes.zig:87).
    pub fn totals(&self) -> (u64, u64, u64, u64, u64) {
        let m = self.state.lock().unwrap();
        (m.prompt, m.generation, m.drafted, m.accepted, m.rounds)
    }

    /// (metrics.zig:97) — the Prometheus text, family for family in the
    /// upstream render's own order, with the vLLM aliases.
    pub fn render(&self, engine: Option<&dyn Engine>, window: u32) -> String {
        let m = self.state.lock().unwrap();
        let mut out = String::with_capacity(8192);
        let mut streams = [0u32; 512];
        let status: Status = engine.map_or(Status::default(), |e| e.status(&mut streams));
        let streams = &streams[..status.streams.min(streams.len())];
        // The engine keeps prompt states: the cached-token families ride it
        let keeps = engine.is_some_and(|e| e.info().prompt_cache);
        // The requests by key and status.
        let mut requests = m.requests.clone();
        requests.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        if !requests.is_empty() {
            family(&mut out, "requests_total", "counter", "HTTP replies by API key label and status.");
            for (key, status, count) in &requests {
                let _ = writeln!(out, "{}requests_total{{key=\"{}\",status=\"{}\"}} {}", PREFIX, key, status, count);
            }
        }
        gauge(&mut out, "requests_running", "gauge", "Requests in prefill or decode.", status.running as u64);
        gauge(&mut out, "requests_waiting", "gauge", "Requests queued or held until a lane is free.", status.waiting as u64);
        gauge(&mut out, "generation_tokens_running", "gauge", "Generated tokens held by live streams.", status.generation_tokens);
        gauge(&mut out, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.", m.prompt);
        if keeps {
            gauge(&mut out, "prompt_tokens_cached_total", "counter", "Prompt tokens of finished requests restored from kept prompt states.", m.cached);
        }
        gauge(&mut out, "generation_tokens_total", "counter", "Generated tokens of finished requests.", m.generation);
        family(&mut out, "kv_cache_usage_ratio", "gauge", "Tokens in a stream cache divided by that stream's context window.");
        pools(&mut out, "kv_cache_usage_ratio", "pool", streams, window);
        gauge(&mut out, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.", m.drafted);
        gauge(&mut out, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.", m.accepted);
        if let Some(h) = &m.latency {
            histogram(&mut out, "request_latency_seconds", "Seconds from arrival to the reply leaving.", h);
        }
        if let Some(h) = &m.ttft {
            histogram(&mut out, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", h);
        }
        if let Some(h) = &m.decode {
            histogram(&mut out, "request_decode_seconds", "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.", h);
        }
        if let Some(h) = &m.tpot {
            histogram(&mut out, "request_time_per_output_token_seconds", "Seconds from the first generated token to the last, divided by the tokens less one.", h);
        }
        gauge(&mut out, "decode_rounds_total", "counter", "Decode rounds used by finished requests.", m.rounds);
        if let Some(h) = &m.prefill {
            histogram(&mut out, "request_prefill_seconds", "Seconds a finished request's prompt pass took.", h);
        }
        if let Some(p) = crate::process_memory::read() {
            gauge(&mut out, "process_footprint_bytes", "gauge", "Current OS physical footprint including Metal buffers (macOS).", p.physical_footprint_bytes);
            gauge(&mut out, "process_footprint_peak_bytes", "gauge", "Kernel lifetime peak physical footprint including Metal buffers; never reset by health (macOS).", p.lifetime_peak_physical_footprint_bytes);
        }
        if let Some(e) = engine {
            if let Some(mem) = e.memory(false) {
                gauge(&mut out, "device_memory_bytes", "gauge", "Device memory the engine holds now: weights, caches and live streams; only where the backend counts it (CUDA).", mem.active);
                gauge(&mut out, "device_memory_peak_bytes", "gauge", "The most device memory the engine has held since start or the last /health?reset_peak=1.", mem.peak);
                gauge(&mut out, "pinned_memory_bytes", "gauge", "Pinned host memory the engine holds now: staging buffers and windows; only where the backend counts it (CUDA).", mem.pinned);
            }
        }
        // The vLLM aliases.
        gauge(&mut out, "num_requests_running", "gauge", "Requests in prefill or decode. A mirror of tensorfold:requests_running.", status.running as u64);
        gauge(&mut out, "num_requests_waiting", "gauge", "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.", status.waiting as u64);
        family(&mut out, "kv_cache_usage_perc", "gauge", "A stream's cache occupancy under vLLM's name; same streams and ratios as tensorfold:kv_cache_usage_ratio.");
        pools(&mut out, "kv_cache_usage_perc", "stream", streams, window);
        gauge(&mut out, "spec_decode_num_draft_tokens_total", "counter", "Draft tokens verified on finished requests, this server's single draft counter.", m.drafted);
        gauge(&mut out, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.", m.accepted);
        if keeps {
            gauge(&mut out, "prefix_cache_queries_total", "counter", "Prompt tokens of finished requests, looked up in the kept prompt states, under vLLM's name.", m.prompt);
            gauge(&mut out, "prefix_cache_hits_total", "counter", "Prompt tokens of finished requests restored from kept prompt states, under vLLM's name.", m.cached);
        }
        if let Some(h) = &m.latency {
            histogram(&mut out, "e2e_request_latency_seconds", "Seconds from arrival to the reply leaving, under vLLM's name.", h);
        }
        if let Some(h) = &m.decode {
            histogram(&mut out, "request_decode_time_seconds", "Seconds a finished request spent decoding, under vLLM's name.", h);
        }
        if let Some(h) = &m.prefill {
            histogram(&mut out, "request_prefill_time_seconds", "Seconds a finished request's prompt pass took, under vLLM's name.", h);
        }
        gauge(&mut out, "client_disconnections_total", "counter", "Requests a client left before the reply left the server.", m.disconnects);
        if let Some(p) = status.preemptions {
            gauge(&mut out, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.", p);
        }
        out
    }
}

/// The per-pool (or per-stream) lines of a kv-cache family (metrics.zig:186):
/// no streams at all still names pool 0.
fn pools(out: &mut String, name: &str, label: &str, lengths: &[u32], window: u32) {
    if lengths.is_empty() {
        let _ = writeln!(out, "{}{}{{{label}=\"0\"}} 0", PREFIX, name);
        return;
    }
    for (i, n) in lengths.iter().enumerate() {
        let ratio: f64 = if window == 0 {
            0.0
        } else {
            (*n as f64 / window as f64).min(1.0)
        };
        let _ = writeln!(out, "{}{}{{{label}=\"{i}\"}} {}", PREFIX, name, format_num(ratio));
    }
}

fn family(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {}{} {}", PREFIX, name, help);
    let _ = writeln!(out, "# TYPE {}{} {}", PREFIX, name, kind);
}

fn gauge(out: &mut String, name: &str, kind: &str, help: &str, value: u64) {
    family(out, name, kind, help);
    let _ = writeln!(out, "{}{} {}", PREFIX, name, value);
}

/// Upstream: `histogram` (metrics.zig:197) — the cumulative bucket counts,
/// the +Inf bucket, the sum and the count.
fn histogram(out: &mut String, name: &str, help: &str, h: &Histogram) {
    let _ = writeln!(out, "# HELP {}{} {}", PREFIX, name, help);
    let _ = writeln!(out, "# TYPE {}{} histogram", PREFIX, name);
    let mut cumulative: u64 = 0;
    for (edge, count) in h.edges.iter().zip(&h.counts) {
        cumulative += count;
        let _ = writeln!(
            out,
            "{}{}_bucket{{le=\"{}\"}} {}",
            PREFIX,
            name,
            format_num(*edge),
            cumulative
        );
    }
    cumulative += h.counts[h.edges.len()];
    let _ = writeln!(out, "{}{}_bucket{{le=\"+Inf\"}} {}", PREFIX, name, cumulative);
    let _ = writeln!(out, "{}{}_sum {}", PREFIX, name, format_num(h.total));
    let _ = writeln!(out, "{}{}_count {}", PREFIX, name, h.n);
}

fn format_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The histogram's cumulative buckets and the +Inf sum.
    #[test]
    fn histogram_buckets_cumulate() {
        let mut h = Histogram::new(&BUCKETS);
        h.observe(0.02); // the 0.05 bucket
        h.observe(1.0); // the 1.0 bucket
        h.observe(1000.0); // +Inf
        let mut out = String::new();
        histogram(&mut out, "test_seconds", "A test.", &h);
        assert!(out.contains("le=\"0.05\"} 1"), "{}", out);
        assert!(out.contains("le=\"1\"} 2"), "{}", out);
        assert!(out.contains("le=\"+Inf\"} 3"), "{}", out);
        assert!(out.contains("test_seconds_count 3"), "{}", out);
    }

    /// The render: the request families and the gauges in the Prometheus text.
    #[test]
    fn render_carries_the_families() {
        let m = Metrics::new();
        m.note(100, 20, 50, 10, 8, 5, 0.5, Some(0.1), Some(0.3), Some(0.1), Some(0.01));
        m.http_request("key1", 200);
        let body = m.render(None, 8192);
        assert!(body.contains("requests_total{key=\"key1\",status=\"200\"} 1"), "{}", body);
        assert!(body.contains("prompt_tokens_total 100"), "{}", body);
        assert!(body.contains("generation_tokens_total 50"), "{}", body);
        assert!(body.contains("mtp_drafted_total 10"), "{}", body);
        assert!(body.contains("request_latency_seconds_count 1"), "{}", body);
        assert!(body.contains("e2e_request_latency"), "{}", body);
        assert!(body.contains("spec_decode_num_draft_tokens_total 10"), "{}", body);
    }
}
