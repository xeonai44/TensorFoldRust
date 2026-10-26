//! The CUDA open's decision arithmetic: env variables, the context
//! window, the weights' bytes, the chip name.
//!
//! Port of `zig/src/native/cuda.zig`'s pure half (cuda.zig:33-98) with
//! upstream's own tests (cuda.zig:413-441). The open itself and the
//! lane-host wiring live in the CLI's serve path (the port's Linux
//! branch), as the Metal family's does on macOS.

use tensorfold_cuda::Driver;

/// The chip class gate entries name ("nvidia-sm121" for a GB10) of the
/// device TF_CUDA_DEVICE picks; None without one. Upstream: `chip`
/// (cuda.zig:33).
pub fn chip() -> Option<String> {
    let ordinal = match env_number(env("TF_CUDA_DEVICE").as_deref()) {
        Ok(o) => o,
        Err(_) => return None,
    };
    let driver = Driver::open().ok()?;
    let count = driver.device_count().ok()?;
    let ordinal = ordinal.unwrap_or(0);
    if ordinal as i32 >= count {
        return None;
    }
    chip_class(driver.capability(ordinal as i32).ok()?)
}

/// Upstream: `chipClass` (cuda.zig:41).
pub fn chip_class(capability: u32) -> Option<String> {
    Some(format!("nvidia-sm{capability}"))
}

fn env(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}

/// A variable's whole number: None when unset or empty, an error for
/// anything else that is not one. Upstream: `envNumber` (cuda.zig:50)
/// — upstream's error{Invalid} is unit; the port names it.
#[derive(Debug, PartialEq)]
pub struct NotAWholeNumber;

impl std::fmt::Display for NotAWholeNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not a whole number")
    }
}

impl std::error::Error for NotAWholeNumber {}

pub fn env_number(text: Option<&str>) -> Result<Option<u32>, NotAWholeNumber> {
    let t = text.unwrap_or("").trim();
    if t.is_empty() {
        return Ok(None);
    }
    t.parse::<u32>().map(Some).map_err(|_| NotAWholeNumber)
}

/// Why the window refused.
#[derive(Debug, PartialEq)]
pub enum WindowError {
    Negative,
    NoNative,
    PastNative,
}

/// Prompt plus reply tokens a request may use: --context (0: the
/// model's window), else the family default within it. Upstream:
/// `contextWindow` (cuda.zig:78).
pub fn context_window(
    requested: Option<i64>,
    native: i64,
    default: i64,
) -> Result<i64, WindowError> {
    let r = match requested {
        None => return Ok(if native > 0 { default.min(native) } else { default }),
        Some(r) => r,
    };
    if r < 0 {
        return Err(WindowError::Negative);
    }
    if r == 0 {
        return if native > 0 { Ok(native) } else { Err(WindowError::NoNative) };
    }
    if native > 0 && r > native {
        return Err(WindowError::PastNative);
    }
    Ok(r)
}

/// The flag's value, else the variable's, else `default`; None with
/// `problem` naming a variable that is not a whole number. Upstream:
/// `setting` (cuda.zig:57).
pub fn setting(
    flag: Option<u32>,
    name: &str,
    default: u32,
    problem: &mut String,
) -> Option<u32> {
    if let Some(v) = flag {
        return Some(v);
    }
    match env_number(env(name).as_deref()) {
        Ok(v) => Some(v.unwrap_or(default)),
        Err(_) => {
            *problem = format!("{name}={}: a whole number", env(name).unwrap_or_default());
            None
        }
    }
}

/// The model's window (config.json's max_position_embeddings,
/// text_config's first), 0 when it names none. Upstream:
/// `modelContext` (cuda.zig:67).
pub fn model_context(dir: &std::path::Path) -> i64 {
    let bytes = match std::fs::read(dir.join("config.json")) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    let doc: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let doc = match doc.as_object() {
        Some(o) => o,
        None => return 0,
    };
    let text = match doc.get("text_config").and_then(|t| t.as_object()) {
        Some(t) => t,
        None => doc,
    };
    let limit = text
        .get("max_position_embeddings")
        .or_else(|| doc.get("max_position_embeddings"));
    match limit.and_then(|v| v.as_i64()) {
        Some(v) if v > 0 => v,
        _ => 0,
    }
}

/// The bytes of the checkpoint's safetensors files: what its weights
/// need on the device, near enough to refuse early. Upstream:
/// `weightBytes` (cuda.zig:87).
pub fn weight_bytes(dir: &std::path::Path) -> u64 {
    let mut total: u64 = 0;
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".safetensors") {
            continue;
        }
        if let Ok(st) = entry.metadata() {
            total += st.len();
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `chip classes name the compute capability as gate
    /// entries do` (cuda.zig:413).
    #[test]
    fn chip_classes_name_the_compute_capability_as_gate_entries_do() {
        assert_eq!(Some("nvidia-sm121".to_string()), chip_class(121));
    }

    /// Upstream's `variables: unset or empty means the default,
    /// anything but a whole number is refused` (cuda.zig:425).
    #[test]
    fn variables_unset_or_empty_means_the_default_anything_but_a_whole_number_is_refused() {
        assert_eq!(Ok(None), env_number(None));
        assert_eq!(Ok(None), env_number(Some("")));
        assert_eq!(Ok(Some(3)), env_number(Some(" 3")));
        assert_eq!(Err(NotAWholeNumber), env_number(Some("one")));
        assert_eq!(Err(NotAWholeNumber), env_number(Some("-1")));
    }

    /// Upstream's `context windows: the family default inside the
    /// model's, 0 for the model's, never past it` (cuda.zig:433).
    #[test]
    fn context_windows_the_family_default_inside_the_models_0_for_the_models_never_past_it() {
        assert_eq!(Ok(16384), context_window(None, 262144, 16384));
        assert_eq!(Ok(4096), context_window(None, 4096, 16384));
        assert_eq!(Ok(262144), context_window(Some(0), 262144, 16384));
        assert_eq!(Ok(65536), context_window(Some(65536), 262144, 16384));
        assert_eq!(
            Err(WindowError::PastNative),
            context_window(Some(300000), 262144, 16384)
        );
        assert_eq!(
            Err(WindowError::Negative),
            context_window(Some(-5), 262144, 16384)
        );
        assert_eq!(Err(WindowError::NoNative), context_window(Some(0), 0, 16384));
    }

    /// The settings' fallback: the flag wins, the variable else the
    /// default, a bad variable refuses with its name (setting,
    /// cuda.zig:57 — upstream tests it only through the open's problem
    /// lines; the port pins the rule).
    #[test]
    fn settings_fall_back_from_flag_to_variable_to_default() {
        let mut problem = String::new();
        assert_eq!(Some(7), setting(Some(7), "TF_TEST_VAR", 1, &mut problem));
        assert!(problem.is_empty());
    }

    /// `modelContext` reads text_config's window first (cuda.zig:67's
    /// rule), else the root's, 0 without one.
    #[test]
    fn model_context_reads_text_config_first_else_the_root() {
        let dir = std::env::temp_dir().join("tf-cuda-model-context");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings": 100, "text_config": {"max_position_embeddings": 200}}"#,
        )
        .unwrap();
        assert_eq!(200, model_context(&dir));
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings": 100}"#,
        )
        .unwrap();
        assert_eq!(100, model_context(&dir));
        std::fs::write(dir.join("config.json"), r#"{"model_type": "x"}"#).unwrap();
        assert_eq!(0, model_context(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
