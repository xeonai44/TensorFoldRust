//! Authenticated, read-only sizing diagnostics; unknown counters and plans
//! are omitted rather than guessed.
//!
//! Port of `zig/src/server/memory_routes.zig` (93 lines).

use crate::state::AppState;
use serde_json::{json, Map, Value};

/// The sizing snapshot: the schema, the window, the process footprint, the
/// device counters, the prompt-cache plan. Upstream: `snapshot`
/// (memory_routes.zig:15) — every unknown reading omitted.
pub fn snapshot(state: &AppState, process: Option<crate::process_memory::Snapshot>) -> Value {
    let mut o = Map::new();
    o.insert("schema_version".into(), json!(1));
    if let Some(info) = state.engine_info.as_ref() {
        if info.context_window > 0 {
            o.insert("context_window".into(), json!(info.context_window));
        }
        o.insert("context_fitted".into(), json!(info.context_fitted));
        if let Some(p) = process {
            o.insert(
                "process".into(),
                json!({
                    "source": "darwin_rusage_info_v4",
                    "physical_footprint_bytes": p.physical_footprint_bytes,
                    "lifetime_peak_physical_footprint_bytes": p.lifetime_peak_physical_footprint_bytes,
                }),
            );
        }
        if let Some(engine) = state.engine.as_ref() {
            if let Some(d) = engine.memory(false) {
                o.insert(
                    "device".into(),
                    json!({
                        "active_bytes": d.active,
                        "cache_bytes": d.cache,
                        "peak_bytes": d.peak,
                        "peak_scope": "since_start_or_health_reset",
                    }),
                );
            }
        }
        if let Some(plan) = info.prompt_cache_plan.as_ref() {
            let mut m = Map::new();
            m.insert("source".into(), json!(plan.source.source_name()));
            m.insert("budget_bytes".into(), json!(plan.budget_bytes));
            m.insert("explicit_budget".into(), json!(plan.explicit_budget));
            if let Some(over) = plan.over_cap {
                m.insert("over_cap".into(), json!(over));
            }
            if let Some(ram) = plan.ram_bytes {
                m.insert("ram_bytes".into(), json!(ram));
            }
            if let Some(ready) = plan.ready_footprint_bytes {
                m.insert("ready_footprint_bytes".into(), json!(ready));
            }
            if let Some(cap) = plan.cap_bytes {
                m.insert("cap_bytes".into(), json!(cap));
            }
            if let Some(room) = plan.room_bytes {
                m.insert("room_bytes".into(), json!(room));
            }
            if let Some(margin) = plan.margin_bytes {
                m.insert("margin_bytes".into(), json!(margin));
            }
            o.insert("prompt_cache".into(), Value::Object(m));
        }
    }
    Value::Object(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The snapshot omits what it does not know (memory_routes.zig:15's
    /// rule, the upstream test's shape).
    #[test]
    fn the_snapshot_omits_the_unknown() {
        let state = AppState::new(crate::Config::new("m".into()));
        let v = snapshot(&state, None);
        assert_eq!(v["schema_version"], 1);
        assert!(v.get("context_window").is_none(), "no info: no window");
        assert!(v.get("process").is_none());
        assert!(v.get("device").is_none());
        assert!(v.get("prompt_cache").is_none());
    }
}
