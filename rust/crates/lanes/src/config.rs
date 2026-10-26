//! The round loop's constants from the model's attributes and the engine's
//! settings (Python _family_setup).
//!
//! Port of `zig/src/core/lanes/config.zig`.

use crate::allocate::extend_costs;
pub use crate::allocate::Cost;
use crate::depth::Constants;
use crate::table::Table;
use crate::Error;

/// The class default acceptance by depth (Python DraftDepth.depth_prior).
/// Upstream: `config.zig:11  pub const default_prior`.
pub const DEFAULT_PRIOR: [f64; 15] = [
    0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
];

/// What a backend reports about its loaded model: the attributes the Python engine
/// reads at setup. Upstream: `config.zig:14  pub const Model` (the family's
/// `facts()` fills this). The field defaults mirror upstream's own
/// (streams_exact and speculate_early true, one draft a round).
#[derive(Clone, Copy)]
pub struct Model<'a> {
    /// The widest window that reproduces one-token steps bit for bit.
    pub exact_width: u32,
    /// A lone stream's first copy window (0: the exact width).
    pub first_copy_rows: u32,
    /// One-token rounds may run a step ahead (pipelined).
    pub gpu_tokens: bool,
    /// A draft head is loaded.
    pub mtp: bool,
    /// The head can draft from a verify's rows.
    pub speculate: bool,
    /// Draft every row's first draft behind the verify.
    pub speculate_early: bool,
    /// The head's acceptance by depth until a stream has its own.
    pub draft_prior: &'a [f64],
    /// Plain rounds compete with drafted depths.
    pub plain_guard: bool,
    /// Head drafts a round at most.
    pub drafts: u32,
    /// Forward ms by exact window width (timed at load).
    pub window_costs: &'a [Cost],
    /// One chained head step (timed at load).
    pub mtp_step_ms: f64,
    /// Shared forwards keep each stream's bits.
    pub streams_exact: bool,
    /// The backend runs several streams' windows in one forward.
    pub hidden_rows: bool,
    /// A shared forward's rows at most (0: 32).
    pub batch_rows: u32,
    /// A shared forward's streams at most (0: 32).
    pub max_streams: u32,
    /// Shared forwards' ms by total rows (timed at load).
    pub shared_costs: &'a [Cost],
    /// The head gives each drafted node's chance of landing.
    pub draft_probabilities: bool,
    /// The head drafts every stream of a shared round in one batch.
    pub draft_streams: bool,
    /// The head drafts trees of lanes (DraftRequest.lanes) and holds them.
    pub head_trees: bool,
    /// A lone stream's window ms by rows, up to its widest tree (timed at load).
    pub lane_costs: &'a [Cost],
}

impl Default for Model<'_> {
    fn default() -> Self {
        Model {
            exact_width: 1,
            first_copy_rows: 0,
            gpu_tokens: false,
            mtp: false,
            speculate: false,
            speculate_early: true,
            draft_prior: &[],
            plain_guard: false,
            drafts: 1,
            window_costs: &[],
            mtp_step_ms: 0.0,
            streams_exact: true,
            hidden_rows: false,
            batch_rows: 0,
            max_streams: 0,
            shared_costs: &[],
            draft_probabilities: false,
            draft_streams: false,
            head_trees: false,
            lane_costs: &[],
        }
    }
}

impl Model<'_> {
    /// The zero-facts model (upstream's field defaults): plain one-row rounds.
    pub fn plain() -> Model<'static> {
        Model::default()
    }
}

/// Upstream: `config.zig:37  pub const Config`.
pub struct Config {
    pub family_width: u32,
    pub max_copy: u32,
    pub first_copy: u32,
    pub base_width: u32,
    pub family_mtp: bool,
    pub speculate_early: bool,
    pub pipelined: bool,
    pub plain_guard: bool,
    pub most_drafts: u32,
    pub family_streams: bool,
    pub batch_rows: u32,
    pub batch_streams: u32,
    pub node_probabilities: bool,
    /// Rows a window may grow to with suffix-match branches (0: off; fill.zig).
    pub fill_lanes: u32,
    /// Context tokens a suffix-match branch must match (shorter ones are
    /// coincidental phrases).
    pub fill_match: u32,
    /// The share of grafted rows a stream must keep to go on reading its chain
    /// back every round.
    pub fill_floor: f64,
    pub draft_streams: bool,
    pub head_trees: bool,
    /// A lone stream's lanes at most, the planner picking each round's (0: the
    /// depth rule's chain).
    pub head_lanes: u32,
    /// And their depth at most.
    pub head_depth: u32,
    /// The planner prices each pick from the stream's own measured rounds (seeded
    /// from the table).
    pub own_prices: bool,
    pub depth_prior: Vec<f64>,
    pub family_costs: Table,
    pub shared_costs: Table,
    /// Window ms by rows to 64 (plan_lanes).
    pub lane_costs: Table,
    pub mtp_step_ms: f64,
    /// Matching tokens a copy needs (rejects coincidental indentation).
    pub enter_match: i64,
    /// A copied token's chance of landing in allocation.
    pub copy_rate: f64,
    pub k: Constants,
}

impl Config {
    /// Upstream: `config.zig:68  pub fn init(gpa, m, max_rows, max_draft)`.
    pub fn new(m: &Model<'_>, max_rows: u32, max_draft: u32) -> Result<Config, Error> {
        if max_rows < 1 {
            return Err(Error::BadMaxRows);
        }
        let width = 1.max(m.exact_width.max(1).min(max_rows));
        let max_copy = width - 1;
        let first_rows: i64 = if m.first_copy_rows != 0 {
            m.first_copy_rows as i64
        } else {
            width as i64
        };
        let first_copy = (1.max(max_copy as i64).min(first_rows - 1)).max(1) as u32;
        let prior: &[f64] = if !m.draft_prior.is_empty() {
            m.draft_prior
        } else {
            &DEFAULT_PRIOR
        };
        let streams = width >= 2 && m.streams_exact && m.hidden_rows;
        let batch_rows: u32 = if m.batch_rows != 0 { m.batch_rows } else { 32 };
        let mut family_costs = Table::new();
        for c in m.window_costs {
            if c.width >= 1 && c.width <= width {
                family_costs.put(c.width as i64, c.ms);
            }
        }
        // Shared forwards: window costs then timed shared rows (a later table
        // overrides), extended to batch_rows.
        let mut timed: Vec<Cost> = Vec::new();
        for table in [m.window_costs, m.shared_costs] {
            for &c in table {
                if c.width < 1 || c.width > batch_rows {
                    continue;
                }
                match timed.iter_mut().find(|t| t.width == c.width) {
                    Some(t) => t.ms = c.ms,
                    None => timed.push(c),
                }
            }
        }
        Ok(Config {
            family_width: width,
            max_copy,
            first_copy,
            base_width: width.min(first_copy + 1),
            family_mtp: m.mtp && width >= 2 && m.speculate,
            speculate_early: m.speculate_early,
            pipelined: m.gpu_tokens,
            plain_guard: m.plain_guard,
            most_drafts: m.drafts.min(width - 1).min(max_draft),
            family_streams: streams,
            batch_rows,
            batch_streams: if streams {
                if m.max_streams != 0 { m.max_streams } else { 32 }
            } else {
                1
            },
            node_probabilities: m.draft_probabilities,
            fill_lanes: 0,
            fill_match: 8,
            fill_floor: 0.35,
            draft_streams: m.draft_streams,
            head_trees: m.head_trees,
            head_lanes: 0,
            head_depth: 16,
            own_prices: true,
            depth_prior: prior.to_vec(),
            family_costs,
            shared_costs: extend_costs(&timed, batch_rows),
            lane_costs: extend_costs(m.lane_costs, 64),
            mtp_step_ms: m.mtp_step_ms,
            enter_match: 8,
            copy_rate: 0.94,
            k: Constants::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors config.zig:130 "setup derives widths like Python".
    #[test]
    fn setup_derives_widths_like_python() {
        let costs: Vec<Cost> = (1..=8)
            .map(|w| Cost { width: w, ms: 5.0 + w as f64 })
            .collect();
        let c = Config::new(
            &Model {
                exact_width: 64,
                first_copy_rows: 16,
                gpu_tokens: true,
                mtp: true,
                speculate: true,
                drafts: 4,
                hidden_rows: true,
                batch_rows: 128,
                max_streams: 64,
                window_costs: &costs,
                ..Model::default()
            },
            64,
            63,
        )
        .unwrap();
        assert_eq!(c.first_copy, 15);
        assert_eq!(c.base_width, 16);
        assert_eq!(c.most_drafts, 4);
        assert!(c.family_mtp && c.family_streams);
    }
}
