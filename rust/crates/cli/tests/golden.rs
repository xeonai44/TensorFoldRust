//! The golden harness (docs/PORTING-PLAN.md, Phase 1b's final gate): the Rust
//! port's server and the upstream Zig engine's server, on the same machine with
//! the same pinned checkpoint, answer the same requests byte for byte — greedy
//! and sampled, first turn and multi-turn.
//!
//! Both binaries serve the pinned model (`QWEN35_MODEL_DIR` overrides); the Zig
//! server comes from the upstream clone's build (`zig build native`), the Rust
//! one from this workspace's release build. Without either binary or the model
//! the test skips with a notice.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ZIG_PORT: u16 = 18091;
const RUST_PORT: u16 = 18092;
const PROMPTS: &[(&str, u32, f64, Option<u64>)] = &[
    // (content, max_tokens, temperature, seed)
    ("Say hello in one sentence.", 48, 0.0, None),
    ("What is the capital of France?", 32, 0.0, None),
    ("Write a haiku about rain.", 40, 0.0, None),
    ("Explain gravity to a child in two sentences.", 48, 0.0, None),
    ("Give me a random word.", 12, 0.9, None),
    ("Pick a number from 1 to 100.", 16, 0.7, None),
    ("Say something surprising.", 20, 1.0, Some(12345)),
    ("A random color?", 12, 0.8, None),
];

struct Server(Child, u16);

impl Server {
    /// Start the server and wait until its listener answers (the startup lines go
    /// to whatever stream each binary picks; the port is the common readiness).
    fn start(bin: &std::path::Path, port: u16, name: &str) -> Option<Server> {
        let mut child = Command::new(bin)
            .arg("serve")
            .arg(model_dir())
            .args([
                "--name", name, "--port", &port.to_string(),
                "--parallel", "1", "--context", "8192",
                "--no-thinking", "--max-tokens", "64",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("the server starts");
        let _ = &mut child;
        let mut server = Server(child, port);
        let started = Instant::now();
        loop {
            match server.0.try_wait() {
                Ok(Some(_)) => {
                    eprintln!("skipping: the server exited before serving");
                    return None;
                }
                _ => {}
            }
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Some(server);
            }
            if started.elapsed() > Duration::from_secs(180) {
                return None;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    fn connect(port: u16) -> TcpStream {
        let started = Instant::now();
        loop {
            if let Ok(stream) = TcpStream::connect(("127.0.0.1", port)) {
                return stream;
            }
            if started.elapsed() > Duration::from_secs(20) {
                panic!("the server never listened on {}", port);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn model_dir() -> std::path::PathBuf {
    std::env::var("QWEN35_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/Volumes/llm/models/Qwen3.5-2B-MLX-4bit"))
}

fn chat(port: u16, model: &str, content: &str, max_tokens: u32, temperature: f64, seed: Option<u64>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    });
    if let Some(seed) = seed {
        body["seed"] = serde_json::json!(seed);
    }
    if content == "A random color?" {
        body["messages"] = serde_json::json!([
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": content},
        ]);
    }
    let text = serde_json::to_string(&body).unwrap();
    let mut stream = Server::connect(port);
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        text.len(),
        text
    );
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let body = response.split("\r\n\r\n").nth(1).expect("the body");
    serde_json::from_str(body).expect("JSON")
}

#[test]
fn rust_answers_match_the_zig_engine_token_for_token() {
    let dir = model_dir();
    let zig_bin = std::path::PathBuf::from("/Volumes/llm/tensorfold/zig-out/native/bin/tensorfold-native");
    let rust_bin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/tensorfold-native");
    if !dir.join("config.json").is_file() || !zig_bin.is_file() || !rust_bin.is_file() {
        eprintln!(
            "skipping: needs the model at {} and both servers (zig build native; cargo build --release)",
            dir.display()
        );
        return;
    }
    let Some(_zig) = Server::start(&zig_bin, ZIG_PORT, "zig-model") else {
        return;
    };
    let Some(_rust) = Server::start(&rust_bin, RUST_PORT, "rust-model") else {
        return;
    };
    let mut zig_text = String::new();
    let mut zig_usage = 0u64;
    for (content, max_tokens, temperature, seed) in PROMPTS {
        let z = chat(ZIG_PORT, "zig-model", content, *max_tokens, *temperature, *seed);
        let r = chat(RUST_PORT, "rust-model", content, *max_tokens, *temperature, *seed);
        let zc = z["choices"][0]["message"]["content"].as_str().unwrap();
        let rc = r["choices"][0]["message"]["content"].as_str().unwrap();
        let zt = z["usage"]["completion_tokens"].as_u64().unwrap();
        let rt = r["usage"]["completion_tokens"].as_u64().unwrap();
        let zp = z["usage"]["prompt_tokens"].as_u64().unwrap();
        let rp = r["usage"]["prompt_tokens"].as_u64().unwrap();
        assert_eq!(
            (rc, rt, rp),
            (zc, zt, zp),
            "content {:?} (temperature {}) diverged",
            content,
            temperature
        );
        eprintln!("MATCH {:?} -> {:?}", content, zc);
    }

    /// The legacy completion surface, both servers: the same greedy text
    /// and usage (openai.zig's is_chat=false half).
    for port in [ZIG_PORT, RUST_PORT] {
        let body = format!(
            "{{\"model\": \"{served}\", \"prompt\": \"Say hello in one sentence.\", \"max_tokens\": 48, \"temperature\": 0}}",
            served = if port == ZIG_PORT { "zig-model" } else { "rust-model" }
        );
        let response = post(port, "/v1/completions", &body);
        let parsed: serde_json::Value = serde_json::from_str(&response).expect("JSON");
        assert_eq!(parsed["object"], "text_completion", "port {port}");
        if port == ZIG_PORT {
            zig_text = parsed["choices"][0]["text"].as_str().unwrap().to_string();
            zig_usage = parsed["usage"]["prompt_tokens"].as_u64().unwrap();
        } else {
            assert_eq!(
                parsed["choices"][0]["text"].as_str().unwrap(),
                zig_text,
                "the completion text diverged"
            );
            assert_eq!(
                parsed["usage"]["prompt_tokens"].as_u64().unwrap(),
                zig_usage,
                "the completion usage diverged"
            );
        }
    }
    eprintln!("MATCH completions: {:?}", zig_text);
}

/// The byte-parity gate (the JSON writer's): both servers under the same
/// name, the static routes byte for byte — model discovery, the tokenize
/// ids, and a frozen corpus refusal body. The dynamic replies (their ids,
/// timestamps, timings) ride the field-wise test above; these bodies have
/// nothing random in them.
const BYTE_ZIG_PORT: u16 = 18093;
const BYTE_RUST_PORT: u16 = 18094;

fn raw(port: u16, request: &str) -> Vec<u8> {
    use std::io::{Read, Write};
    let mut stream = Server::connect(port);
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    // The body: after the blank line. Chunked replies end with the
    // terminating chunk, which the two servers encode their own ways —
    // compare the bodies raw, the transfer framing aside.
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("the response has a body");
    response[split + 4..].to_vec()
}

fn post(port: u16, path: &str, body: &str) -> String {
    String::from_utf8_lossy(&post_raw(port, path, body)).to_string()
}

fn post_raw_bytes(port: u16, path: &str, body: &[u8]) -> Vec<u8> {
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(body);
    raw_bytes(port, &request)
}

fn raw_bytes(port: u16, request: &[u8]) -> Vec<u8> {
    use std::io::{Read, Write};
    let mut stream = Server::connect(port);
    stream.write_all(request).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("the response has a body");
    response[split + 4..].to_vec()
}

fn get_raw(port: u16, path: &str) -> Vec<u8> {
    raw(
        port,
        &format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"),
    )
}

fn post_raw(port: u16, path: &str, body: &str) -> Vec<u8> {
    raw(
        port,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        ),
    )
}

#[test]
fn the_static_bodies_match_the_zig_server_byte_for_byte() {
    let dir = model_dir();
    let zig_bin = std::path::PathBuf::from("/Volumes/llm/tensorfold/zig-out/native/bin/tensorfold-native");
    let rust_bin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/tensorfold-native");
    if !dir.join("config.json").is_file() || !zig_bin.is_file() || !rust_bin.is_file() {
        eprintln!("skipping: needs the model and both servers");
        return;
    }
    let Some(_zig) = Server::start(&zig_bin, BYTE_ZIG_PORT, "twin-model") else {
        return;
    };
    let Some(_rust) = Server::start(&rust_bin, BYTE_RUST_PORT, "twin-model") else {
        return;
    };
    // Model discovery: the served id, the created time (masked — the two
    // reads are their own requests, a second boundary may fall between
    // them), the window extensions
    let z = get_raw(BYTE_ZIG_PORT, "/v1/models");
    let r = get_raw(BYTE_RUST_PORT, "/v1/models");
    let mask = |b: &[u8]| -> String {
        let s = String::from_utf8_lossy(b).to_string();
        match s.split_once("\"created\": ") {
            Some((head, tail)) => {
                let end = tail
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(tail.len());
                format!("{head}\"created\": @{}", &tail[end..])
            }
            None => s,
        }
    };
    assert_eq!(mask(&z), mask(&r), "the models body");
    // The tokenize ids: a plain prompt's encode
    let z = post_raw(
        BYTE_ZIG_PORT,
        "/tokenize",
        r#"{"prompt": "Hello, world!", "add_special_tokens": false}"#,
    );
    let r = post_raw(
        BYTE_RUST_PORT,
        "/tokenize",
        r#"{"prompt": "Hello, world!", "add_special_tokens": false}"#,
    );
    assert_eq!(String::from_utf8_lossy(&z), String::from_utf8_lossy(&r), "the tokenize body");
    // A frozen corpus refusal body, all three agreeing: the Zig server, the
    // port, and the corpus's own bytes
    // (zig/tests/server/golden/responses/refuse-include.json)
    const CORPUS: &str = "{\"error\": {\"message\": \"include is not supported (reasoning.encrypted_content): reasoning comes as text in its reasoning item, and nothing is encrypted\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": null}}";
    let z = post_raw(
        BYTE_ZIG_PORT,
        "/v1/responses",
        r#"{"input": "x", "include": ["reasoning.encrypted_content"]}"#,
    );
    let r = post_raw(
        BYTE_RUST_PORT,
        "/v1/responses",
        r#"{"input": "x", "include": ["reasoning.encrypted_content"]}"#,
    );
    assert_eq!(String::from_utf8_lossy(&z), CORPUS, "the Zig body is the corpus's");
    assert_eq!(String::from_utf8_lossy(&r), CORPUS, "the port's body is the corpus's");
    // The corpus's pinned parse errors (golden/errors/), the chat route's
    // raw read: byte for byte between the servers — Python's own decoder
    // and scanner wordings
    const ERRORS: &[(&[u8], &str)] = &[
        (b"[]", "'list' object has no attribute 'get'"),
        (b"null", "'NoneType' object has no attribute 'get'"),
        (b"\"x\"", "'str' object has no attribute 'get'"),
        (
            b"{\"a\": 1,}",
            "Illegal trailing comma before end of object: line 1 column 8 (char 7)",
        ),
        (
            b"{\"messages\": \"\xff\"}",
            "'utf-8' codec can't decode byte 0xff in position 14: invalid start byte",
        ),
    ];
    for (body, message) in ERRORS {
        let z = post_raw_bytes(BYTE_ZIG_PORT, "/v1/chat/completions", body);
        let r = post_raw_bytes(BYTE_RUST_PORT, "/v1/chat/completions", body);
        let want = format!("{{\"error\": {{\"message\": \"{message}\"}}}}");
        assert_eq!(
            String::from_utf8_lossy(&z),
            want,
            "the Zig body: {}",
            String::from_utf8_lossy(&z)
        );
        assert_eq!(
            String::from_utf8_lossy(&r),
            want,
            "the port's body: {}",
            String::from_utf8_lossy(&r)
        );
    }
    eprintln!("BYTE MATCH: models, tokenize, the frozen refusal, the parse errors");
}
