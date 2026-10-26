//! The round loop on the fake target: drafted == one-token rounds, shared rounds ==
//! solo, greedy and sampled.
//!
//! Port of `zig/src/core/lanes/engine_test.zig` — every case, in the same order.
//! These are the lane core's exactness gates: the tokens a stream emits must not
//! depend on how the round loop scheduled them.

use tensorfold_engine::Sampling;
use tensorfold_lanes::config::{Config, Cost, Model};
use tensorfold_lanes::engine::Engine;
use tensorfold_lanes::fake::{self, Fake};
use tensorfold_lanes::proposer::{Params, SuffixLookup};
use tensorfold_lanes::stream::Spec;
use tensorfold_lanes::FixedClock;

#[derive(Clone)]
struct Case<'a> {
    prompt: &'a [u32],
    max_new: u32,
    sampling: Option<Sampling>,
    drafts: bool,
    think_budget: u32,
    logprobs: Option<u8>,
}

impl<'a> Case<'a> {
    fn new(prompt: &'a [u32]) -> Case<'a> {
        Case {
            prompt,
            max_new: 40,
            sampling: None,
            drafts: true,
            think_budget: 0,
            logprobs: None,
        }
    }
}

fn model() -> Config {
    let costs: Vec<Cost> = (1..=16)
        .map(|w| Cost { width: w, ms: 5.0 + 0.8 * w as f64 })
        .collect();
    Config::new(
        &Model {
            exact_width: 16,
            gpu_tokens: true,
            mtp: true,
            speculate: true,
            speculate_early: false,
            drafts: 4,
            window_costs: &costs,
            mtp_step_ms: 0.5,
            hidden_rows: true,
            batch_rows: 32,
            max_streams: 8,
            draft_streams: true,
            ..Model::default()
        },
        16,
        15,
    )
    .unwrap()
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: &[Case<'_>]) -> Vec<Vec<u32>> {
    let cfg = model();
    let mut target = Fake::new();
    let mut clock = FixedClock::default();
    let mut engine = Engine::init(&cfg, &mut target, &mut clock);
    let mut streams: Vec<Box<tensorfold_lanes::Stream<'_>>> = Vec::with_capacity(cases.len());
    let mut proposers: Vec<Box<SuffixLookup>> = Vec::with_capacity(cases.len());
    for c in cases {
        // Upstream keeps a parallel proposer array (engine_test.zig:38); the
        // stream owns its own, so the spare exists to be dropped with them.
        let proposer = Box::new(SuffixLookup::new(Params { min_match: 4, ..Params::default() }).unwrap());
        let stream = Box::new(tensorfold_lanes::Stream::init(Spec {
            id: "s",
            prompt: c.prompt,
            max_new: c.max_new,
            eos: &[96],
            sampling: c.sampling,
            drafts: c.drafts,
            proposer: Some(Box::new(SuffixLookup::new(Params { min_match: 4, ..Params::default() }).unwrap())),
            think_budget: c.think_budget,
            think_close: &[90, 91, 92],
            think_end: 91,
            logprobs: c.logprobs,
            ..Spec::new("s", c.prompt, c.max_new)
        }));
        proposers.push(proposer);
        streams.push(stream);
    }
    for i in 0..streams.len() {
        unsafe { engine.add_stream(&mut *streams[i]) }.unwrap();
    }
    while engine.active_count() > 0 {
        engine.step().unwrap();
    }
    for (c, s) in cases.iter().zip(streams.iter()) {
        if let Some(k) = c.logprobs {
            expect_rows(c, s, k);
        }
    }
    streams.iter().map(|s| s.emitted().to_vec()).collect()
}

/// Each emitted token's row is the fake target's after the prompt and the tokens
/// before it (forced: by `forToken`).
fn expect_rows(c: &Case<'_>, s: &tensorfold_lanes::Stream<'_>, k: u8) {
    assert_eq!(s.emitted().len(), s.rows.len());
    let mut history: Vec<u32> = c.prompt.to_vec();
    for (i, (&t, got)) in s.emitted().iter().zip(s.rows.iter()).enumerate() {
        let want = fake::row_at(
            &history,
            fake::next(&history, c.sampling, history.len() as u64),
            k,
        )
        .for_token(t);
        assert_eq!(want.token, got.token, "token {}'s row", i);
        assert_eq!(want.logprob.to_bits(), got.logprob.to_bits(), "token {}'s logprob", i);
        assert_eq!(&want.ids[..k as usize], &got.ids[..got.count as usize], "token {}'s ids", i);
        history.push(t);
    }
}

#[derive(Clone)]
struct GuardCase<'a> {
    prompt: &'a [u32],
    max_new: u32,
    drafts: bool,
    think_budget: u32,
    loop_guard: bool,
    think_open: Option<bool>,
    cycle_after: usize,
    pattern: &'a [u32],
    answer_cycles: bool,
}

struct GuardOut {
    emitted: Vec<u32>,
    reason: tensorfold_lanes::Reason,
    period: Option<u32>,
    finished: bool,
}

fn run_guard(cases: &[GuardCase<'_>], step_limit: usize) -> Vec<GuardOut> {
    let cfg = model();
    let c = &cases[0];
    let mut target = Fake::new();
    target.cycle_after = Some(c.cycle_after);
    target.probe_prompt = c.prompt.len();
    target.pattern = c.pattern.to_vec();
    target.answer_cycles = c.answer_cycles;
    let mut clock = FixedClock::default();
    let mut engine = Engine::init(&cfg, &mut target, &mut clock);
    let mut streams: Vec<Box<tensorfold_lanes::Stream<'_>>> = Vec::new();
    for item in cases {
        let stream = Box::new(tensorfold_lanes::Stream::init(Spec {
            id: "guard",
            prompt: item.prompt,
            max_new: item.max_new,
            eos: &[96],
            drafts: item.drafts,
            think_budget: item.think_budget,
            think_close: &[90, 91, 92],
            think_end: 91,
            think_open: item.think_open,
            loop_guard: item.loop_guard,
            ..Spec::new("guard", item.prompt, item.max_new)
        }));
        streams.push(stream);
    }
    for i in 0..streams.len() {
        unsafe { engine.add_stream(&mut *streams[i]) }.unwrap();
    }
    let mut steps = 0;
    while engine.active_count() > 0 && steps < step_limit {
        engine.step().unwrap();
        steps += 1;
    }
    streams
        .iter()
        .map(|s| GuardOut {
            emitted: s.emitted().to_vec(),
            reason: s.reason,
            period: s.loop_period,
            finished: s.finished,
        })
        .collect()
}

const P1: [u32; 11] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
const P2: [u32; 13] = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9];

/// Mirrors engine_test.zig:132: drafted rounds commit the one-token decode, greedy
/// and sampled.
#[test]
fn drafted_rounds_commit_the_one_token_decode() {
    for s in [None, Some(Sampling { seed: 5, temperature: 0.7, top_k: 0, top_p: 0.95, ..Sampling::default() })] {
        let drafted = run(&[Case { sampling: s, ..Case::new(&P1) }]);
        let plain = run(&[Case { sampling: s, drafts: false, ..Case::new(&P1) }]);
        assert_eq!(plain[0], drafted[0]);
        // And the fake target's own decode.
        let mut history: Vec<u32> = P1.to_vec();
        for &t in &drafted[0] {
            assert_eq!(fake::next(&history, s, history.len() as u64), t);
            history.push(t);
        }
    }
}

/// Mirrors engine_test.zig:150: shared rounds commit what each stream commits
/// alone.
#[test]
fn shared_rounds_commit_what_each_stream_commits_alone() {
    let sampled = Sampling { seed: 9, temperature: 1.0, top_k: 0, top_p: 0.9, ..Sampling::default() };
    let together = run(&[
        Case::new(&P1),
        Case { prompt: &P2, sampling: Some(sampled), max_new: 30, ..Case::new(&P2) },
    ]);
    let one = run(&[Case::new(&P1)]);
    let two = run(&[Case { prompt: &P2, sampling: Some(sampled), max_new: 30, ..Case::new(&P2) }]);
    assert_eq!(one[0], together[0]);
    assert_eq!(two[0], together[1]);
}

/// Mirrors engine_test.zig:162: the thinking budget's forced close is the same
/// drafted, plain and shared.
#[test]
fn the_thinking_budgets_forced_close_is_the_same_every_path() {
    let drafted = run(&[Case { think_budget: 9, ..Case::new(&P2) }]);
    let plain = run(&[Case { think_budget: 9, drafts: false, ..Case::new(&P2) }]);
    let shared = run(&[
        Case { think_budget: 9, ..Case::new(&P2) },
        Case { drafts: false, ..Case::new(&P1) },
    ]);
    assert_eq!(plain[0], drafted[0]);
    assert_eq!(plain[0], shared[0]);
    assert_eq!(drafted[0][8], 90);
}

/// Mirrors engine_test.zig:174: logprob rows follow the committed tokens drafted,
/// plain, shared, sampled and through a forced close.
#[test]
fn logprob_rows_follow_the_committed_tokens() {
    let sampled = Sampling { seed: 5, temperature: 0.7, top_k: 0, top_p: 0.95, ..Sampling::default() };
    let cases_sets: Vec<Vec<Vec<Case<'_>>>> = vec![
        vec![vec![Case { logprobs: Some(3), ..Case::new(&P1) }]],
        vec![vec![Case { logprobs: Some(3), drafts: false, ..Case::new(&P1) }]],
        vec![vec![Case { logprobs: Some(0), sampling: Some(sampled), ..Case::new(&P1) }]],
        vec![vec![
            Case { logprobs: Some(20), ..Case::new(&P1) },
            Case { prompt: &P2, sampling: Some(sampled), max_new: 30, logprobs: Some(2), ..Case::new(&P2) },
        ]],
        vec![vec![Case { think_budget: 9, logprobs: Some(4), ..Case::new(&P2) }]],
    ];
    for with in cases_sets {
        let without: Vec<Case<'_>> = with[0]
            .iter()
            .map(|c| Case {
                prompt: c.prompt,
                max_new: c.max_new,
                sampling: c.sampling,
                drafts: c.drafts,
                think_budget: c.think_budget,
                ..Case::new(c.prompt)
            })
            .collect();
        let a = run(&with[0]); // checks every row
        let b = run(&without);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(y, x);
        }
    }
}

/// Mirrors engine_test.zig:194: the loop guard matches drafted and plain, reports
/// its period, and answers after close.
#[test]
fn the_loop_guard_matches_drafted_and_plain() {
    let plain = run_guard(
        &[GuardCase { drafts: false, ..guard_base(&P2) }],
        5000,
    );
    let drafted = run_guard(&[guard_base(&P2)], 5000);
    assert_eq!(plain[0].emitted, drafted[0].emitted);
    assert_eq!(drafted[0].period, Some(3));
    assert!(plain[0].finished && plain[0].reason == tensorfold_lanes::Reason::Stop);
    assert_eq!(&plain[0].emitted[329..], &[90, 91, 92, 40, 41, 42, 96]);

    let off = run_guard(
        &[GuardCase { loop_guard: false, think_open: Some(true), ..guard_base(&P2) }],
        5000,
    );
    assert!(off[0].finished && off[0].reason == tensorfold_lanes::Reason::Length && off[0].emitted.len() == 400);
}

/// Mirrors engine_test.zig:209: a cycle in the answer does not refire or reclose.
#[test]
fn a_cycle_in_the_answer_does_not_refire_or_reclose() {
    let plain = run_guard(
        &[GuardCase { drafts: false, max_new: 800, answer_cycles: true, ..guard_base(&P2) }],
        5000,
    );
    let drafted = run_guard(
        &[GuardCase { max_new: 800, answer_cycles: true, ..guard_base(&P2) }],
        5000,
    );
    assert_eq!(plain[0].emitted, drafted[0].emitted);
    assert_eq!(plain[0].period, Some(3));
    assert!(plain[0].finished && plain[0].reason == tensorfold_lanes::Reason::Length);
    let close_count = plain[0].emitted.iter().filter(|&&t| t == 90).count();
    assert_eq!(close_count, 1);
}

fn guard_base(prompt: &[u32]) -> GuardCase<'_> {
    GuardCase {
        prompt,
        max_new: 400,
        drafts: true,
        think_budget: 0,
        loop_guard: true,
        think_open: None,
        cycle_after: 70,
        pattern: &[11, 12, 13],
        answer_cycles: false,
    }
}
