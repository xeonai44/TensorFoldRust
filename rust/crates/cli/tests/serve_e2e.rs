//! The end-to-end gate (docs/PORTING-PLAN.md, Phase 1b part 2's exit): the
//! `tensorfold-native` binary serves the pinned checkpoint and answers
//! `/v1/chat/completions` with a real chat completion — the template rendered,
//! the lane round loop driving the Metal backend, the reply decoded.
//!
//! The test drives the release binary against the pinned model directory
//! (`QWEN35_MODEL_DIR` overrides), on its own port; without a model or a Metal
//! device it skips with a notice, the way upstream's GPU tests do.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 18087;

fn model_dir() -> std::path::PathBuf {
    std::env::var("QWEN35_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/Volumes/llm/models/Qwen3.5-2B-MLX-4bit"))
}

struct Server(Child);

impl Server {
    fn start() -> Option<Server> {
        let dir = model_dir();
        if !dir.join("config.json").is_file() {
            eprintln!("skipping: no model at {} (set QWEN35_MODEL_DIR)", dir.display());
            return None;
        }
        let bin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/tensorfold-native");
        if !bin.is_file() {
            eprintln!("skipping: no release binary; cargo build --release first");
            return None;
        }
        let child = Command::new(bin)
            .arg("serve")
            .arg(&dir)
            .args([
                "--name",
                "local-model",
                "--port",
                &PORT.to_string(),
                "--parallel",
                "2",
                "--context",
                "8192",
                "--no-thinking",
                "--max-tokens",
                "64",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("the server starts");
        let mut server = Server(child);
        // The startup line means the model loaded and the listener is up —
        // upstream's ``[tensorfold]`` log goes to stdout.
        let started = Instant::now();
        let stdout = server.0.stdout.take().unwrap();
        let mut lines = BufReader::new(stdout).lines();
        while started.elapsed() < Duration::from_secs(120) {
            match lines.next() {
                Some(Ok(line)) => {
                    eprintln!("serve: {}", line);
                    if line.contains("serving local-model at") {
                        return Some(server);
                    }
                }
                _ => break,
            }
        }
        None
    }

    fn connect() -> TcpStream {
        // The startup line arrives just before the bind; retry until it listens.
        let started = Instant::now();
        loop {
            if let Ok(stream) = TcpStream::connect(("127.0.0.1", PORT)) {
                return stream;
            }
            if started.elapsed() > Duration::from_secs(10) {
                panic!("the server never listened on {}", PORT);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn get(&self, path: &str) -> String {
        let mut stream = Self::connect();
        stream
            .write_all(format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").as_bytes())
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    fn delete(&self, path: &str) -> String {
        let mut stream = Self::connect();
        stream
            .write_all(format!("DELETE {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").as_bytes())
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    fn post(&self, body: &str) -> String {
        self.post_at("/v1/chat/completions", body)
    }

    fn post_at(&self, path: &str, body: &str) -> String {
        let mut stream = Self::connect();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A second server with compaction forced (every request over the tiny
/// fraction's threshold rebuilds), on its own port.
fn start_compacting() -> Option<Server> {
    let dir = model_dir();
    if !dir.join("config.json").is_file() {
        return None;
    }
    let bin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/tensorfold-native");
    if !bin.is_file() {
        return None;
    }
    let memory = std::env::temp_dir().join(format!(
        "tf-compact-e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let child = Command::new(bin)
        .arg("serve")
        .arg(&dir)
        .args([
            "--name", "local-model",
            "--port", "18088",
            "--parallel", "2",
            "--no-thinking",
            "--max-tokens", "64",
            "--compact-at", "0.001",
            "--compact-keep", "50",
            "--compact-memory",
        ])
        .arg(&memory)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("the compacting server starts");
    let mut server = Server(child);
    let started = Instant::now();
    let stdout = server.0.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    while started.elapsed() < Duration::from_secs(120) {
        match lines.next() {
            Some(Ok(line)) => {
                eprintln!("compacting serve: {}", line);
                if line.contains("serving local-model at") {
                    return Some(server);
                }
            }
            _ => break,
        }
    }
    None
}

fn json_body(response: &str) -> serde_json::Value {
    let body = response
        .split("\r\n\r\n")
        .nth(1)
        .expect("the response has a body");
    serde_json::from_str(body).expect("the body is JSON")
}

#[test]
fn serve_answers_chat_completions_on_real_weights() {
    let Some(server) = Server::start() else {
        return;
    };
    // A greedy reply: real text, a stop or length finish, and usage counts that
    // carry the rendered prompt's tokens.
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":48,"temperature":0}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let reply = json_body(&response);
    assert_eq!(reply["model"], "local-model");
    let content = reply["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty(), "the reply has text");
    let prompt_tokens = reply["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(prompt_tokens > 0, "the prompt tokens count");
    let completion = reply["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(completion > 0, "the completion tokens count");
    eprintln!("greedy reply: {:?}", content);

    // The streamed reply: OpenAI chunks, ending in [DONE]. The prose
    // chunks carry {"delta": {"content": ...}} — the wire shape the
    // Messages and Responses machines read (openai.zig:267's wrap), in
    // the json.dumps spacing every body now carries.
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Count: one two"}],"max_tokens":16,"temperature":0,"stream":true}"#,
    );
    assert!(response.contains("chat.completion.chunk"));
    assert!(response.contains(r#""delta": {"content": "#));
    assert!(response.contains("[DONE]"));

    // The Anthropic Messages stream (anthropic.zig's Wire over the same
    // chunks): message_start, the typed block deltas, message_stop.
    let response = server.post_at(
        "/v1/messages",
        r#"{"model":"local-model","messages":[{"role":"user","content":"Count: one two"}],"max_tokens":16,"stream":true}"#,
    );
    assert!(response.contains("message_start"), "{}", response);
    assert!(response.contains("content_block_start"), "{}", response);
    assert!(response.contains("content_block_delta"), "{}", response);
    assert!(response.contains("text_delta"), "{}", response);
    assert!(response.contains("message_stop"), "{}", response);

    // logprobs: the Metal lanes have no first-row producer, so the server
    // refuses them exactly as upstream does on Metal (the CUDA lanes are
    // the backend that reports them; the refusal is the pinned model's
    // behavior, openai.zig:124's exact_bytes gate).
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"temperature":0,"logprobs":true,"top_logprobs":3}"#,
    );
    assert!(response.starts_with("HTTP/1.1 400"), "{}", response);
    assert!(
        response.contains("logprobs are not supported by this model or backend"),
        "{}",
        response
    );
    // The streamed request refuses the same way (the gate runs before the
    // streamed-only exclusion).
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"temperature":0,"stream":true,"logprobs":true}"#,
    );
    assert!(
        response.contains("logprobs are not supported by this model or backend"),
        "{}",
        response
    );

    // The thinking budget: the engine forces the think close at the budget,
    // the reasoning tokens bounded by it (chat.zig:198-204's budget through
    // the engine's close forcing).
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Think step by step, then answer: what is 2+2?"}],"max_tokens":96,"temperature":0,"thinking_budget":8}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let reply = json_body(&response);
    let reasoning = reply["usage"]["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .expect("the reasoning tokens count");
    assert!(reasoning <= 12, "the budget bounds the thinking: {}", reasoning);
    eprintln!("budgeted reasoning tokens: {}", reasoning);

    // Structured output: no engine in this tree reports structures, so the
    // server refuses the request exactly as upstream does (server.zig:181).
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"temperature":0,"response_format":{"type":"json_object"}}"#,
    );
    assert!(response.starts_with("HTTP/1.1 400"), "{}", response);
    assert!(
        response.contains("structured output (response_format and the guided_* fields) is not supported by this engine yet"),
        "{}",
        response
    );
    // A malformed one refuses with the grammar's own words, before the
    // engine check
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"temperature":0,"response_format":"json_object"}"#,
    );
    assert!(
        response.contains("response_format must be an object"),
        "{}",
        response
    );

    // Model discovery carries the window.
    let mut stream = Server::connect();
    stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let models = json_body(&response);
    assert_eq!(models["data"][0]["id"], "local-model");
    assert!(models["data"][0].get("context_length").is_some());

    // Compaction: a second server with the fraction forced low and a small
    // keep, a multi-turn conversation rebuilt as a note and its kept tail
    // (compact.zig's run through the real engine — the note itself a
    // greedy generation).
    if let Some(compacting) = start_compacting() {
        let mut conversation = String::from(
            r#"{"model":"local-model","messages":["#,
        );
        for i in 0..20 {
            conversation.push_str(&format!(
                r#"{{"role":"user","content":"Turn {i}: tell me about topic number {i} in some detail please."}},"#
            ));
            conversation.push_str(&format!(
                r#"{{"role":"assistant","content":"Here is a detailed answer about topic {i} with various facts."}},"#
            ));
        }
        conversation.push_str(
            r#"{"role":"user","content":"Summarize everything so far."}],"max_tokens":64,"temperature":0}"#,
        );
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", 18088)).unwrap();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            conversation.len(),
            conversation
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
        let reply = json_body(&response);
        let runtime = &reply["tensorfold"];
        assert!(
            runtime["context_window"].as_u64().is_some_and(|w| w > 0),
            "the window stamped"
        );
        assert!(
            runtime["context_used"].as_u64().is_some(),
            "the prompt's use stamped"
        );
        let compaction = &runtime["compaction"];
        assert!(
            compaction.is_object(),
            "the reply compacted: {}",
            serde_json::to_string(runtime).unwrap_or_default()
        );
        assert!(
            compaction["messages_compacted"].as_u64().is_some_and(|n| n > 0),
            "the conversation compacted"
        );
        assert!(
            compaction["kept_messages"].as_u64().is_some_and(|n| n < 40),
            "the kept tail small"
        );
        assert!(
            !compaction["note"].as_str().unwrap_or_default().is_empty(),
            "the note the model wrote"
        );
        assert!(
            compaction["note_tokens"].as_u64().is_some_and(|n| n > 0),
            "the note's own generation counted"
        );
        let content = reply["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(!content.is_empty(), "the reply answers past its note");
        eprintln!(
            "compacted: {} messages into a {}-token note",
            compaction["messages_compacted"], compaction["note_tokens"]
        );
        drop(compacting);
    }

    // The pinned family serves cache-less (native/qwen35.zig arms no
    // store — the tranche-twenty-six fix): a repeated long conversation's
    // rerun reports no cached tokens, exactly as the Zig server does, and
    // the greedy reply stays identical.
    let filler = "You are a meticulous assistant. ".repeat(700); // ~10,500 tokens
    let conv = format!(
        r#"{{"model":"local-model","messages":[{{"role":"system","content":{}}},{{"role":"user","content":{}}},{{"role":"assistant","content":{}}},{{"role":"user","content":"What number should you remember?"}}],"max_tokens":64,"temperature":0}}"#,
        serde_json::to_string(&filler).unwrap(),
        serde_json::to_string(&format!("Remember the number 42. {}", &filler[..4000])).unwrap(),
        serde_json::to_string(&format!("I will remember the number 42. {}", &filler[..2000])).unwrap(),
    );
    let first = server.post(&conv);
    assert!(first.starts_with("HTTP/1.1 200"), "{}", first);
    let first_reply = json_body(&first);
    let first_cached = first_reply["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let first_content = first_reply["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    eprintln!("first request: {} cached tokens, reply {:?}", first_cached, &first_content[..first_content.len().min(40)]);
    // A rerun of the same conversation: the kept state serves it
    let second = server.post(&conv);
    assert!(second.starts_with("HTTP/1.1 200"), "{}", second);
    let second_reply = json_body(&second);
    let second_cached = second_reply["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let second_content = second_reply["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    eprintln!("rerun: {} cached tokens (first held {})", second_cached, first_cached);
    // Upstream's qwen3_5 host keeps no prompt states: both runs report
    // none (the store machinery waits for the family that arms one)
    assert_eq!(
        (first_cached, second_cached),
        (0, 0),
        "the pinned family serves cache-less, exactly as upstream"
    );
    // The greedy reply stays identical across runs
    assert_eq!(first_content, second_content, "the replies are the same");

    // The legacy completions surface (openai.zig's is_chat=false half):
    // the prompt as the model reads it, the reply a text_completion.
    let response = server.post_at(
        "/v1/completions",
        r#"{"model":"local-model","prompt":"Say hello in one sentence.","max_tokens":48,"temperature":0}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let reply = json_body(&response);
    assert_eq!(reply["object"], "text_completion");
    let id = reply["id"].as_str().unwrap();
    assert!(id.starts_with("cmpl-"), "{}", id);
    let text = reply["choices"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty(), "the completion carries text");
    assert_eq!(reply["choices"][0]["logprobs"], serde_json::json!(null));
    assert_eq!(
        reply["usage"]["total_tokens"].as_u64().unwrap(),
        reply["usage"]["prompt_tokens"].as_u64().unwrap()
            + reply["usage"]["completion_tokens"].as_u64().unwrap()
    );
    assert!(reply["tensorfold"].is_object(), "the runtime rides");
    // The empty prompts: the render's own refusal (the corpus's null and
    // missing cases)
    for body in ["{\"prompt\": null}", "{}"] {
        let response = server.post_at("/v1/completions", body);
        assert!(response.starts_with("HTTP/1.1 400"), "{}", response);
        assert!(response.contains("rendered prompt is empty"), "{}", response);
    }
    // Bad ids: the range refusal naming the vocabulary's own bound
    let response = server.post_at("/v1/completions", r#"{"prompt": [5, 99999999]}"#);
    assert!(response.starts_with("HTTP/1.1 400"), "{}", response);
    assert!(
        response.contains("prompt token ids must be integers in the valid range 0 to"),
        "{}",
        response
    );
    // Bools reach the ids branch and refuse on the value (the corpus's
    // bool-ids case)
    let response = server.post_at("/v1/completions", r#"{"prompt": [true, 1]}"#);
    assert!(response.starts_with("HTTP/1.1 400"), "{}", response);
    assert!(
        response.contains("prompt token ids must be integers in the valid range 0 to"),
        "{}",
        response
    );
    // A number reads as its Python text (the corpus's number case)
    let response = server.post_at("/v1/completions", r#"{"prompt": 12, "max_tokens": 8}"#);
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    // The nested strings join on newlines (the corpus's nested case)
    let response = server.post_at(
        "/v1/completions",
        r#"{"prompt": [["Say hi."], "In one word: yes"], "max_tokens": 12}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    // Messages ride the template (the corpus's messages case)
    let response = server.post_at(
        "/v1/completions",
        r#"{"model":"local-model","messages":[{"role":"user","content":"Say hi."}],"max_tokens":12,"temperature":0}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    assert_eq!(json_body(&response)["object"], "text_completion");
    // logprobs: the model's own refusal first (the Metal engine reports
    // no rows — the same refusal the chat route gives); the
    // completions-only wording ("logprobs are supported on
    // /v1/chat/completions only") fires on an engine that reports them
    let response = server.post_at(
        "/v1/completions",
        r#"{"prompt":"hi","logprobs":true,"top_logprobs":3}"#,
    );
    assert!(response.starts_with("HTTP/1.1 400"), "{}", response);
    assert!(
        response.contains("logprobs are not supported by this model or backend"),
        "{}",
        response
    );
    // The streamed completion: text chunks, the finish, [DONE]
    let response = server.post_at(
        "/v1/completions",
        r#"{"model":"local-model","prompt":"Count: one two","max_tokens":16,"temperature":0,"stream":true}"#,
    );
    assert!(response.contains(r#""object": "text_completion""#), "{}", response);
    assert!(response.contains(r#""text": ""#), "{}", response);
    assert!(response.contains("[DONE]"), "{}", response);
    // The usage a chunk of its own (the corpus's usage-chunk case)
    let response = server.post_at(
        "/v1/completions",
        r#"{"model":"local-model","prompt":"Count: one two","max_tokens":16,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert!(response.contains(r#""choices": [], "usage""#), "{}", response);
    assert!(response.contains("[DONE]"), "{}", response);
    // The chat stream's own usage chunk (the same mechanism, the chat side)
    let response = server.post(
        r#"{"model":"local-model","messages":[{"role":"user","content":"Count: one two"}],"max_tokens":16,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert!(response.contains(r#""choices": [], "usage""#), "{}", response);
    assert!(response.contains("[DONE]"), "{}", response);

    // The Responses API (responses.zig): each request runs as its chat
    // completion, translated item by item — the whole response object, the
    // stored chain, the streamed typed events.
    let response = server.post_at(
        "/v1/responses",
        r#"{"model":"local-model","input":"Say hello in one sentence.","max_output_tokens":48}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let reply = json_body(&response);
    assert_eq!(reply["object"], "response");
    assert_eq!(reply["status"], "completed");
    assert_eq!(reply["model"], "local-model");
    assert_eq!(reply["incomplete_details"], serde_json::json!(null));
    let id = reply["id"].as_str().expect("the response id").to_string();
    assert!(id.starts_with("resp_"), "{}", id);
    let output = reply["output"].as_array().expect("the output items");
    let message = output
        .iter()
        .find(|item| item["type"] == "message")
        .expect("a message item");
    assert!(
        !message["content"][0]["text"].as_str().unwrap_or_default().is_empty(),
        "the message carries text"
    );
    assert!(
        reply["usage"]["input_tokens"].as_u64().is_some_and(|n| n > 0),
        "the usage counts: {}",
        response
    );
    assert_eq!(reply["usage"]["total_tokens"].as_u64().unwrap(),
        reply["usage"]["input_tokens"].as_u64().unwrap()
            + reply["usage"]["output_tokens"].as_u64().unwrap());
    assert_eq!(reply["store"], serde_json::json!(true));
    assert_eq!(reply["truncation"], "disabled");
    assert!(reply["completed_at"].as_u64().is_some());
    assert!(reply["tensorfold"].is_object(), "the runtime rides");
    // The stored chain: a follow-up with previous_response_id runs the
    // whole conversation; GET returns it; DELETE removes it
    let follow = server.post_at(
        "/v1/responses",
        &serde_json::json!({
            "model": "local-model",
            "input": "And what is your name?",
            "previous_response_id": id,
            "max_output_tokens": 48,
        })
        .to_string(),
    );
    assert!(follow.starts_with("HTTP/1.1 200"), "{}", follow);
    let follow_reply = json_body(&follow);
    let second_id = follow_reply["id"].as_str().unwrap().to_string();
    assert_eq!(follow_reply["previous_response_id"], serde_json::json!(id));
    assert!(
        follow_reply["usage"]["input_tokens"].as_u64().unwrap()
            > reply["usage"]["input_tokens"].as_u64().unwrap(),
        "the follow-up carries the stored conversation's tokens"
    );
    let stored = server.get(&format!("/v1/responses/{id}"));
    assert!(stored.starts_with("HTTP/1.1 200"), "{}", stored);
    assert_eq!(json_body(&stored)["id"], serde_json::json!(id));
    let deleted = server.delete(&format!("/v1/responses/{id}"));
    assert!(deleted.starts_with("HTTP/1.1 200"), "{}", deleted);
    assert_eq!(
        json_body(&deleted),
        serde_json::json!({"id": id, "object": "response", "deleted": true})
    );
    let gone = server.get(&format!("/v1/responses/{id}"));
    assert!(gone.starts_with("HTTP/1.1 404"), "{}", gone);
    assert!(
        gone.contains("no stored response has id"),
        "the 404 names it: {}",
        gone
    );
    // A follow-up naming the deleted parent: the exact refusal
    let orphan = server.post_at(
        "/v1/responses",
        &serde_json::json!({
            "model": "local-model",
            "input": "x",
            "previous_response_id": id,
        })
        .to_string(),
    );
    assert!(orphan.starts_with("HTTP/1.1 400"), "{}", orphan);
    assert!(
        orphan.contains("is not stored here (store: false, deleted, or evicted)"),
        "{}",
        orphan
    );
    // The streamed reply: the typed events in order, ending completed
    let streamed = server.post_at(
        "/v1/responses",
        r#"{"model":"local-model","input":"Count: one two","max_output_tokens":16,"stream":true}"#,
    );
    assert!(streamed.contains("event: response.created"), "{}", streamed);
    assert!(streamed.contains("response.in_progress"), "{}", streamed);
    assert!(streamed.contains("response.output_item.added"), "{}", streamed);
    assert!(streamed.contains("response.output_text.delta"), "{}", streamed);
    assert!(streamed.contains("response.completed"), "{}", streamed);
    assert!(
        streamed.contains("\"sequence_number\""),
        "the events count: {}",
        streamed
    );
    // The translate refusals: the corpus's own words with this route's
    // error shape
    let refused = server.post_at(
        "/v1/responses",
        r#"{"input":"x","include":["reasoning.encrypted_content"]}"#,
    );
    assert!(refused.starts_with("HTTP/1.1 400"), "{}", refused);
    // The whole body, byte for byte the frozen corpus's own
    // (zig/tests/server/golden/responses/refuse-include.json)
    const CORPUS: &str = "{\"error\": {\"message\": \"include is not supported (reasoning.encrypted_content): reasoning comes as text in its reasoning item, and nothing is encrypted\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": null}}";
    assert!(refused.contains(CORPUS), "{}", refused);
    let _ = second_id;

    // The token routes (tokens.zig:25 + :82): /tokenize gives the ids the
    // chat route would run — a plain prompt's encode, or the chat
    // template's render on messages — and /detokenize round-trips the
    // ids back to text.
    let response = server.post_at("/tokenize", r#"{"prompt":"hi","add_special_tokens":false}"#);
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let tokens = json_body(&response);
    assert!(tokens["count"].as_u64().is_some_and(|n| n > 0), "{}", response);
    assert!(
        tokens["max_model_len"].as_u64().is_some_and(|n| n > 0),
        "the window carries: {}",
        response
    );
    let ids = tokens["tokens"].as_array().expect("the ids").clone();
    assert_eq!(ids.len() as u64, tokens["count"].as_u64().unwrap());
    // Round trip: the ids decode back to the prompt
    let body = serde_json::json!({"tokens": ids}).to_string();
    let response = server.post_at("/detokenize", &body);
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    assert_eq!(json_body(&response)["prompt"].as_str().unwrap(), "hi");
    // The messages shape renders the chat template (the ids the chat route
    // would run)
    let response = server.post_at(
        "/tokenize",
        r#"{"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":true}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let chat_ids = json_body(&response);
    assert!(
        chat_ids["count"].as_u64().is_some_and(|n| n > ids.len() as u64),
        "the template renders more than the bare encode: {}",
        response
    );
    // The token texts come beside the ids when asked
    let response = server.post_at(
        "/tokenize",
        r#"{"prompt":"hi","add_special_tokens":false,"return_token_strs":true}"#,
    );
    let with_strs = json_body(&response);
    assert!(
        with_strs["token_strs"]
            .as_array()
            .is_some_and(|a| !a.is_empty()),
        "{}",
        response
    );
    // A flagged special id detokenizes as its own text (hf_text.zig:233's
    // keep-specials decode)
    let eos = chat_ids["tokens"]
        .as_array()
        .and_then(|a| a.last())
        .cloned()
        .unwrap_or(serde_json::json!(0));
    let response = server.post_at("/detokenize", &serde_json::json!({"tokens": [eos]}).to_string());
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);

    // The client-disconnect cancellation (the corpus's cancel group,
    // chat.zig:423's Generation.cancel): a streaming client that leaves
    // mid-reply cancels the engine's request, the disconnection counted.
    use std::io::Read as _;
    {
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", PORT)).unwrap();
        let body = r#"{"model":"local-model","messages":[{"role":"user","content":"Write a long story about a robot."}],"max_tokens":64,"temperature":0,"stream":true}"#;
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).unwrap();
        // Read until a few events arrived, then leave mid-reply
        let mut seen = String::new();
        let started = std::time::Instant::now();
        while seen.matches("data:").count() < 3 && started.elapsed() < std::time::Duration::from_secs(30) {
            let mut chunk = [0u8; 512];
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => seen.push_str(&String::from_utf8_lossy(&chunk[..n])),
            }
        }
        drop(stream); // the client leaves mid-reply
        // The engine's cancellation lands, the disconnection counted
        std::thread::sleep(std::time::Duration::from_secs(3));
        let response = server.get("/metrics");
        let count = response
            .lines()
            .find_map(|l| l.strip_prefix("tensorfold:client_disconnections_total "))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        assert!(count >= 1, "the disconnection counted: {}", count);
        eprintln!("client left mid-reply; disconnections: {count}");
        // A fresh request works: the lane the cancellation freed
        let response = server.post(
            r#"{"model":"local-model","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8,"temperature":0}"#,
        );
        assert!(response.starts_with("HTTP/1.1 200"), "the lane freed: {}", response);
    }

    // The status routes: /health carries the live line's counts, /v1/memory
    // the sizing snapshot (status_routes.zig:13 + memory_routes.zig:15).
    let response = server.get("/health");
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let health = json_body(&response);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["model"], "local-model");
    assert!(health["live"]["connections"].is_u64());
    let response = server.get("/v1/memory");
    assert!(response.starts_with("HTTP/1.1 200"), "{}", response);
    let memory = json_body(&response);
    assert_eq!(memory["schema_version"], 1);
    assert!(memory["context_window"].as_u64().is_some());
    // The pinned family arms no store: the plan's section stays absent,
    // upstream's own shape
    assert!(memory.get("prompt_cache").is_none(), "{}", response);
    assert!(memory["process"]["physical_footprint_bytes"].as_u64().is_some());

    drop(server);
}
