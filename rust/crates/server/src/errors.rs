//! A request's refusal: Python's RequestError family, or another exception's
//! bare message.
//!
//! Port of `zig/src/server/errors.zig` (62 lines): the refusal kinds and
//! their statuses (errors.zig:5's Kind, errors.zig:46's status), the context
//! limit's wording, and the error bodies the OpenAI surface writes
//! (openai.zig:86's errorBody, openai.zig:98's wrapError).

use serde_json::{json, Value};

/// Upstream: `Kind` (errors.zig:5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// RequestError: 400, `{"message", "type": "invalid_request_error"}`.
    Request,
    /// ContextLengthError: as request, plus `param` and
    /// `code: context_length_exceeded`.
    ContextLength,
    /// CapacityError: 503, retry shortly.
    Capacity,
    /// Any other exception (ValueError, KeyError...): 400 with the message
    /// only.
    Other,
    /// An exception while the reply runs: 500 with the message, or a
    /// stream's server_error event.
    Server,
}

/// OpenAI's wording, which clients match to compact. Upstream:
/// `context_limit` (errors.zig:18).
pub const CONTEXT_LIMIT: &str = "This server's maximum context length is";

/// The request's refusal: the kind and the message. Upstream: errors.zig:22's
/// `Cx` (the allocator aside, which the port's message owns).
#[derive(Clone, Debug)]
pub struct Refused {
    pub kind: Kind,
    pub message: String,
}

impl Refused {
    /// Upstream: `refuse` (errors.zig:34).
    pub fn request(message: impl Into<String>) -> Refused {
        Refused {
            kind: Kind::Request,
            message: message.into(),
        }
    }

    /// Upstream: `fail(.context_length, ...)` (errors.zig:28).
    pub fn context_length(message: impl Into<String>) -> Refused {
        Refused {
            kind: Kind::ContextLength,
            message: message.into(),
        }
    }

    /// Upstream: `fail(.capacity, ...)` (errors.zig:28).
    pub fn capacity(message: impl Into<String>) -> Refused {
        Refused {
            kind: Kind::Capacity,
            message: message.into(),
        }
    }

    /// Upstream: `fail(.server, ...)` (errors.zig:28).
    pub fn server(message: impl Into<String>) -> Refused {
        Refused {
            kind: Kind::Server,
            message: message.into(),
        }
    }

    /// Upstream: `other` (errors.zig:40).
    pub fn other(message: impl Into<String>) -> Refused {
        Refused {
            kind: Kind::Other,
            message: message.into(),
        }
    }

    /// Upstream: `status` (errors.zig:46).
    pub fn status(&self) -> u16 {
        match self.kind {
            Kind::Capacity => 503,
            Kind::Server => 500,
            _ => 400,
        }
    }
}

impl From<Refused> for String {
    fn from(r: Refused) -> String {
        r.message
    }
}

/// `errorBody` (openai.zig:86): the message; the type for the request
/// family; `param` and `code` for the context refusal. `param` is the field
/// the refusal blames — "messages" for the chat surface, "prompt" for the
/// completions one (openai.zig:219).
pub fn error_body(refused: &Refused, param: &str) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("message".into(), json!(refused.message));
    if matches!(refused.kind, Kind::Other | Kind::Server) {
        return Value::Object(o);
    }
    o.insert("type".into(), json!("invalid_request_error"));
    if refused.kind == Kind::ContextLength {
        o.insert("param".into(), json!(param));
        o.insert("code".into(), json!("context_length_exceeded"));
    }
    Value::Object(o)
}

/// `wrapError` (openai.zig:98): the error body under `"error"`.
pub fn wrap_error(body: Value) -> Value {
    json!({"error": body})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kinds_statuses_and_bodies() {
        // request: 400 with the type
        let r = Refused::request("stop must be a nonempty string");
        assert_eq!(r.status(), 400);
        assert_eq!(
            error_body(&r, "messages"),
            json!({"message": "stop must be a nonempty string", "type": "invalid_request_error"})
        );
        // context_length: the type plus param and code
        let r = Refused::context_length("too long");
        assert_eq!(r.status(), 400);
        assert_eq!(
            error_body(&r, "messages"),
            json!({
                "message": "too long", "type": "invalid_request_error",
                "param": "messages", "code": "context_length_exceeded",
            })
        );
        // capacity: 503 with the type
        let r = Refused::capacity("the engine is busy; retry shortly");
        assert_eq!(r.status(), 503);
        assert!(error_body(&r, "messages").get("code").is_none());
        // other and server: the message alone
        let r = Refused::other("the scheduler is closed");
        assert_eq!(r.status(), 400);
        assert_eq!(error_body(&r, "messages"), json!({"message": "the scheduler is closed"}));
        let r = Refused {
            kind: Kind::Server,
            message: "the reply failed".into(),
        };
        assert_eq!(r.status(), 500);
        assert_eq!(error_body(&r, "messages"), json!({"message": "the reply failed"}));
        // wrapped
        assert_eq!(
            wrap_error(error_body(&Refused::request("x"), "messages")),
            json!({"error": {"message": "x", "type": "invalid_request_error"}})
        );
    }
}
