//! Python's repr of a JSON value, for refusal messages that quote a shape,
//! and the Python semantics the tool surfaces lean on: truthiness and
//! ``repr(float)``.
//!
//! Port of `zig/src/server/pyrepr.zig` (104 lines) at the fidelity the port's
//! error messages need: the scalar reprs exact, containers readable. The
//! truthiness and float repr come from their own upstream files, cited below.

use serde_json::Value;

/// Python truthiness: None, False, 0, 0.0, "", [] and {} are false.
///
/// Upstream: `truthy` (json.zig:47).
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else {
                n.as_f64().is_none_or(|f| f != 0.0)
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// ``repr(float)``: the shortest round-trip digits, fixed between 1e-4 and
/// 1e16, else exponent form with a signed two-digit-or-more exponent.
///
/// Upstream: `floatRepr` (json_write.zig:104) — the digits from the
/// shortest scientific rendering, trailing zeros trimmed, rearranged by the
/// decimal point's place.
pub fn float_repr(f: f64) -> String {
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // The shortest round-trip scientific form: [-]d[.ddd]e[-]X.
    let sci = format!("{:e}", f);
    let (mantissa, exp_str) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp_str.parse().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let mut digits: String = mantissa
        .trim_start_matches('-')
        .chars()
        .filter(|c| *c != '.')
        .collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let n = digits.len();
    let decpt = exp + 1;
    let mut out = String::with_capacity(24);
    if negative {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let mag = exp.unsigned_abs();
        if mag < 10 {
            out.push('0');
        }
        out.push_str(&mag.to_string());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let d = decpt as usize;
        if d >= n {
            out.push_str(&digits);
            for _ in 0..d - n {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..d]);
            out.push('.');
            out.push_str(&digits[d..]);
        }
    }
    out
}

/// Upstream: `repr` (pyrepr.zig) — Python's repr of the value.
pub fn repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(o) => {
            let parts: Vec<String> = o
                .iter()
                .map(|(k, v)| format!("{}: {}", repr(&Value::String(k.clone())), repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalars_and_containers_quote_like_python() {
        assert_eq!(repr(&Value::Null), "None");
        assert_eq!(repr(&json!(true)), "True");
        assert_eq!(repr(&json!(1.5)), "1.5");
        assert_eq!(repr(&json!("a'b")), "'a\\'b'");
        assert_eq!(repr(&json!([1, "x"])), "[1, 'x']");
    }

    #[test]
    fn python_truthiness() {
        assert!(!truthy(&Value::Null));
        assert!(!truthy(&json!(false)));
        assert!(!truthy(&json!(0)));
        assert!(!truthy(&json!(0.0)));
        assert!(!truthy(&json!("")));
        assert!(!truthy(&json!([])));
        assert!(!truthy(&json!({})));
        assert!(truthy(&json!(true)));
        assert!(truthy(&json!(1)));
        assert!(truthy(&json!(-0.5)));
        assert!(truthy(&json!("x")));
        assert!(truthy(&json!([0])));
        assert!(truthy(&json!({"a": null})));
    }

    #[test]
    fn float_repr_matches_python() {
        assert_eq!(float_repr(0.0), "0.0");
        assert_eq!(float_repr(-0.0), "-0.0");
        assert_eq!(float_repr(5.0), "5.0");
        assert_eq!(float_repr(5.5), "5.5");
        assert_eq!(float_repr(100.0), "100.0");
        assert_eq!(float_repr(0.0001), "0.0001");
        // 1e-5 sits on the exponent side of Python's repr cut
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(1e15), "1000000000000000.0");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1e30), "1e+30");
        assert_eq!(float_repr(-2.5e-8), "-2.5e-08");
        assert_eq!(float_repr(1.234e-4), "0.0001234");
        assert_eq!(float_repr(0.1 + 0.2), "0.30000000000000004");
    }
}

/// The think opener upstream's decision route searches for (decisions.zig:
/// 350's literal) — `reply_text::THINK_OPEN`, not the channel markers'
/// own open (which the "" spelling would leave matching everywhere).
pub fn think_open_marker() -> &'static str {
    crate::reply_text::THINK_OPEN
}
