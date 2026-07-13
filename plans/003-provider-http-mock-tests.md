# Plan 003: Add an HTTP-mock layer and deterministic per-provider request/response tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index (if `plans/README.md` does not exist, skip this).
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- Cargo.toml src/model_client/gemini.rs src/model_client/groq.rs tests/provider_contract_tests.rs`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/002-ci-verification-gates.md
- **Category**: tests
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

The five provider clients in `src/model_client/` (~1460 lines: openai, anthropic, gemini, groq, bedrock) have zero deterministic test coverage. The only tests that exercise them are `tests/model_client_tests.rs`, which require live API keys, cost money, and silently skip when no keys are present — so a regression in request framing (e.g. the `response_format.json_schema` block OpenAI requires for structured output) or response parsing would ship undetected. This plan adds `wiremock` as a dev-dependency and a new keyless test file, `tests/provider_contract_tests.rs`, that pins down (1) request-body shape per provider, (2) response parsing from canned JSON (happy + error paths), (3) non-2xx → `ModelClientError::Http` mapping, and (4) positional alignment of results in a mixed success/failure batch. After this lands, `cargo test` catches provider-contract regressions with no network credentials, and the CI gate added by plans/002-ci-verification-gates.md runs these tests on every PR.

## Current state

### Files and roles

- `src/model_client/mod.rs` — `ModelClient` trait; shared HTTP client (`pub fn http_client()` at lines 29–31); default `send_request_structured` with the non-2xx → `Http` mapping (lines 159–183); default `format_request_body` with per-provider structured-output framing (lines 186–228); private `run_batch` (lines 397–417) reached through the public wrappers `fetch_data_generic` (lines 433–439, errors → `None`) and `fetch_data_generic_with_schema` (lines 442–450, errors → JSON error objects).
- `src/model_client/openai.rs` — OpenAI client. Already honors `OPENAI_BASE_URL` (lines 57–61). Overrides `format_request_body` (lines 88–112). `OpenAIMessage.content` is `Option<String>` (line 30), so `"content": null` parses to `None` and `parse_response` (lines 114–122) returns `ParseError("No response content")`.
- `src/model_client/anthropic.rs` — Anthropic client. Already honors `ANTHROPIC_BASE_URL` (lines 80–84). Overrides `apply_auth` (lines 90–94: `x-api-key` + `anthropic-version: 2023-06-01`), `format_request_body` (lines 119–177: top-level `system`, `max_tokens: 4096`, forced-tool-use `tools`/`tool_choice`), `parse_response` (lines 179–200: tool_use input first, then text), and **its own `send_request_structured`** (lines 202–235) — so Anthropic's non-2xx → `Http` mapping lives at anthropic.rs:230–234, not in mod.rs.
- `src/model_client/gemini.rs` — Gemini client. `api_endpoint` is **hardcoded** (lines 62–64) — no base-URL override exists yet. Auth via `x-goog-api-key` header (lines 70–75). `format_request_body` (lines 98–123) emits `contents`, optional `system_instruction`, and `generationConfig` with `response_mime_type`/`response_json_schema`. `parse_response` (lines 125–134) returns `ParseError("No response content")` on empty `candidates`.
- `src/model_client/groq.rs` — Groq client. `api_endpoint` is **hardcoded** (lines 56–58) — no override exists yet. OpenAI-compatible: `format_request_body` (lines 84–105) emits `response_format.json_schema`; `GroqMessage.content` is `Option<String>` (line 29); `parse_response` at lines 107–115.
- `src/model_client/bedrock.rs` — Bedrock client. Uses the AWS SDK (`aws_sdk_bedrockruntime`), not reqwest — **it cannot be wiremocked** and is limited in this plan to no-network request-construction assertions on its public surface (`format_messages` lines 128–141, `format_request_body` lines 143–148, `api_endpoint` lines 119–122, `get_api_key` lines 203–206, `with_region` lines 55–58). Its message-conversion helper `convert_messages_to_bedrock` is private — do not make it public.
- `tests/model_client_tests.rs` — existing live-API integration tests. Lines 16–20 are the repo's exemplar for serializing tests that share process-global state:

  ```rust
  // tests/model_client_tests.rs:16-20
  static NET_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

  fn network_guard() -> std::sync::MutexGuard<'static, ()> {
      NET_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
  }
  ```

- `Cargo.toml` — `[dev-dependencies]` currently contains only `dotenvy = "0.15"` and `tokio-test = "0.4"` (lines 46–48). `tokio` already has `rt-multi-thread` + `macros` features (line 25), so `#[tokio::test]` works in test files. Edition is 2021 (line 4), so `std::env::set_var` is a safe function here.

### Key excerpts (verified against the working tree at `afc78da`)

Non-2xx mapping in the trait default, `src/model_client/mod.rs:175-183`:

```rust
        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }
```

Structured-output framing in the trait default, `src/model_client/mod.rs:196-219` (OpenAI/Groq `response_format.json_schema`; Anthropic `tools` + `tool_choice`):

```rust
                match self.provider() {
                    Provider::OpenAI | Provider::Groq => {
                        // OpenAI and Groq use response_format with json_schema
                        body["response_format"] = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "name": model_name.unwrap_or("response"),
                                "strict": true,
                                "schema": schema_value
                            }
                        });
                    },
                    Provider::Anthropic => {
                        // Anthropic uses forced tool use for structured outputs
                        body["tools"] = serde_json::json!([{
                            "name": model_name.unwrap_or("response"),
                            "description": "Extract structured data according to the schema",
                            "input_schema": schema_value
                        }]);
                        body["tool_choice"] = serde_json::json!({
                            "type": "tool",
                            "name": model_name.unwrap_or("response")
                        });
                    },
```

The existing base-URL override pattern to copy, `src/model_client/openai.rs:57-61`:

```rust
    fn api_endpoint(&self) -> String {
        let base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        format!("{}/v1/chat/completions", base.trim_end_matches('/'))
    }
```

The two hardcoded endpoints you will make overridable:

```rust
// src/model_client/gemini.rs:62-64
    fn api_endpoint(&self) -> String {
        format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model)
    }
```

```rust
// src/model_client/groq.rs:56-58
    fn api_endpoint(&self) -> String {
        "https://api.groq.com/openai/v1/chat/completions".to_string()
    }
```

Batch alignment machinery, `src/model_client/mod.rs:397-417` (`run_batch` is private; test it through the public `fetch_data_generic` at mod.rs:433-439, which maps a failed row to `None` at index N without shifting other rows — see `run_one` at mod.rs:367-395):

```rust
async fn run_batch<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>,
    errors_as_json: bool,
) -> Vec<Option<String>> {
    let schema_check = SchemaCheck::compile(schema);
    ...
    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}
```

### Conventions and constraints

- Error handling: provider errors are `ModelClientError` variants (`Http(u16, String)`, `Serialization`, `RequestError`, `ParseError(String)`) — mod.rs:92-98. Assert variants with `matches!`.
- Env-var mutation in tests is process-global; the repo's pattern is a poisoning-tolerant static mutex held for the whole test body (`tests/model_client_tests.rs:16-20`, quoted above). Match it.
- `OPENAI_BASE_URL` is also used by the local MLX server backend (README.md:231, tests/test_local_server_backend.py) — the two new env vars must follow the exact same semantics (base host only; client appends the path; `trim_end_matches('/')`).
- Do NOT call `dotenvy::dotenv()` in the new test file — loading a developer's `.env` could inject real keys/base-URLs into the process.
- Documented tradeoffs you must not disturb: pyo3 pinned <0.29 (`.cargo/audit.toml`), reqwest rustls-native-roots / ring choices (Cargo.toml comments). Adding a dev-dependency does not touch these.
- Commit style: sentence-case imperative summaries, e.g. `Refresh cargo-audit ignore rationale for the pyo3 CVEs` (from `git log`).
- CI currently runs **no** `cargo test` (only tarpaulin with `|| true` in the coverage job). plans/002-ci-verification-gates.md adds the keyless `cargo test` gate; this plan's tests are what that gate will protect. This plan can be implemented and verified locally before 002 lands, but its CI value arrives with 002.

## Commands you will need

Run from `/Users/daviddrummond/SideProjects/polar-llama` (all verified in this repo; none need API keys):

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile lib + tests | `cargo check --tests` | exit 0 |
| Format check | `cargo fmt --all -- --check` | exit 0, no output |
| Lint (repo standard) | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0 |
| New tests only, guaranteed keyless | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` | `test result: ok.` all pass, 0 failed |
| Unit tests (regression) | `cargo test --lib` | `test result: ok.` |
| Full keyless suite | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY cargo test` | exit 0 (live tests in `model_client_tests.rs` self-skip without keys; note they call `dotenvy::dotenv()`, so if a `.env` with real keys exists they will hit live APIs — that is pre-existing behavior, not this plan's concern) |

Verified during planning: `cargo test --test model_client_tests --no-run` compiles and links in this repo (pyo3 `extension-module` does not break integration-test binaries here), and `wiremock` latest on crates.io is `0.6.5` (async, tokio-compatible).

## Scope

**In scope** (the only files you may modify/create):

- `Cargo.toml` — add `wiremock = "0.6"` to `[dev-dependencies]` only.
- `src/model_client/gemini.rs` — the `api_endpoint` body only (add `GEMINI_BASE_URL` override).
- `src/model_client/groq.rs` — the `api_endpoint` body only (add `GROQ_BASE_URL` override).
- `tests/provider_contract_tests.rs` — create.
- `plans/README.md` — status row only, if the file exists.

**Out of scope** (do NOT touch, even though they look related):

- `src/model_client/openai.rs` and `src/model_client/anthropic.rs` — they already honor `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL`; no source change needed or allowed.
- `src/model_client/mod.rs` and `src/model_client/bedrock.rs` — no behavior changes; do not make private items (`run_batch`, `convert_messages_to_bedrock`) public to test them.
- Any change to request/response logic, retry behavior (that is plans/008), or the shape of any request body.
- `tests/model_client_tests.rs`, all Python tests, `.github/workflows/*` (CI wiring is plans/002).
- `Cargo.toml` `[dependencies]` — wiremock must be a dev-dependency only; the shipped wheel must not gain runtime deps.

## Git workflow

- Branch: `advisor/003-provider-http-mock-tests` (branched from `main`).
- One commit per step or logical unit; sentence-case imperative summaries, e.g. `Add GEMINI_BASE_URL and GROQ_BASE_URL endpoint overrides` and `Add keyless provider contract tests with wiremock`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add wiremock as a dev-dependency

In `Cargo.toml`, extend `[dev-dependencies]` (currently lines 46–48):

```toml
[dev-dependencies]
dotenvy = "0.15"
tokio-test = "0.4"
# HTTP mocking for keyless provider contract tests (tests/provider_contract_tests.rs)
wiremock = "0.6"
```

**Verify**: `cargo check --tests` → exit 0 (downloads and compiles wiremock).

### Step 2: Add GEMINI_BASE_URL and GROQ_BASE_URL overrides

Copy the `OPENAI_BASE_URL` pattern (openai.rs:57-61, quoted in "Current state") exactly. Note Anthropic already has its override — do not touch it.

Replace `src/model_client/gemini.rs:62-64` with:

```rust
    fn api_endpoint(&self) -> String {
        // GEMINI_BASE_URL overrides the host for proxies/tests (mirrors OPENAI_BASE_URL).
        let base = std::env::var("GEMINI_BASE_URL")
            .unwrap_or_else(|_| "https://generativelanguage.googleapis.com".to_string());
        format!(
            "{}/v1beta/models/{}:generateContent",
            base.trim_end_matches('/'),
            self.model
        )
    }
```

Replace `src/model_client/groq.rs:56-58` with:

```rust
    fn api_endpoint(&self) -> String {
        // GROQ_BASE_URL overrides the host for proxies/tests (mirrors OPENAI_BASE_URL).
        let base = std::env::var("GROQ_BASE_URL")
            .unwrap_or_else(|_| "https://api.groq.com".to_string());
        format!("{}/openai/v1/chat/completions", base.trim_end_matches('/'))
    }
```

No other lines in either file change. Auth flow (`apply_auth`, `get_api_key`) is untouched.

**Verify**: `cargo check` → exit 0, and
`grep -n "GEMINI_BASE_URL" src/model_client/gemini.rs && grep -n "GROQ_BASE_URL" src/model_client/groq.rs` → one hit each.

### Step 3: Create tests/provider_contract_tests.rs — scaffolding, request-shape, and parse tests (no network)

Create `tests/provider_contract_tests.rs`. File header, imports, env lock, and shared fixtures:

```rust
//! Deterministic, keyless contract tests for the HTTP provider clients.
//! No live API keys are required; HTTP interactions run against wiremock.
//! Bedrock uses the AWS SDK (not reqwest) and cannot be wiremocked, so it is
//! covered only by no-network request-construction assertions.

use polar_llama::model_client::{
    create_client, fetch_data_generic, http_client, Message, ModelClient, ModelClientError,
    Provider,
};
use polar_llama::model_client::anthropic::AnthropicClient;
use polar_llama::model_client::bedrock::BedrockClient;
use polar_llama::model_client::gemini::GeminiClient;
use polar_llama::model_client::groq::GroqClient;
use polar_llama::model_client::openai::OpenAIClient;
use serde_json::{json, Value};
use wiremock::matchers::{body_partial_json, body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Serializes tests that read or mutate process-global *_BASE_URL env vars.
/// Same pattern as NET_TEST_LOCK in tests/model_client_tests.rs:16-20 —
/// poisoning is ignored so one failing test doesn't cascade.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn user_msg(content: &str) -> Message {
    Message { role: "user".to_string(), content: content.to_string(), cache_control: None }
}

fn system_msg(content: &str) -> Message {
    Message { role: "system".to_string(), content: content.to_string(), cache_control: None }
}

const SCHEMA: &str = r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
```

Important: every test that starts a `MockServer` and sets a `*_BASE_URL` var MUST take `let _env = env_guard();` as its first statement and keep it for the whole body (env vars are process-global; the default multi-threaded test runner would otherwise race). Pure shape/parse tests below do not touch env vars and need no guard. Do NOT call `dotenvy::dotenv()` anywhere in this file.

Then add the no-network tests. Request-shape (call `format_request_body` directly via the `ModelClient` trait):

1. `openai_request_body_includes_json_schema_response_format` — `OpenAIClient::new_with_model("gpt-4o-mini")`, `format_request_body(&[user_msg("hi")], Some(SCHEMA), Some("my_schema"))`; assert `body["model"] == "gpt-4o-mini"`, `body["messages"][0]["role"] == "user"`, `body["response_format"]["type"] == "json_schema"`, `body["response_format"]["json_schema"]["name"] == "my_schema"`, `body["response_format"]["json_schema"]["strict"] == true`, and `body["response_format"]["json_schema"]["schema"]["required"][0] == "answer"`. Also assert `body.get("temperature").is_none()` (the client deliberately sends no temperature — openai.rs:89-91 comment).
2. `groq_request_body_includes_json_schema_response_format` — same assertions against `GroqClient::new_with_model("llama-3.3-70b-versatile")` (groq.rs:84-105 mirrors OpenAI).
3. `anthropic_request_body_uses_forced_tool_choice` — `AnthropicClient::new_with_model("claude-haiku-4-5")`, messages `[system_msg("be brief"), user_msg("hi")]`, schema `SCHEMA`, model_name `Some("my_schema")`; assert `body["max_tokens"] == 4096`, `body["system"] == "be brief"` (plain string when no cache_control — anthropic.rs:152-157), `body["messages"]` has length 1 (system extracted), `body["tools"][0]["name"] == "my_schema"`, `body["tools"][0]["input_schema"]["required"][0] == "answer"`, `body["tool_choice"]["type"] == "tool"`, `body["tool_choice"]["name"] == "my_schema"`.
4. `gemini_request_body_uses_generation_config_schema` — `GeminiClient::new_with_model("gemini-2.5-flash")`, messages `[system_msg("be brief"), user_msg("hi"), Message{role:"assistant",..}]`; assert `body["system_instruction"]["parts"][0]["text"] == "be brief"`, `body["contents"]` contains no `"system"` role, assistant maps to role `"model"` (gemini.rs:84-87), `body["generationConfig"]["response_mime_type"] == "application/json"`, `body["generationConfig"]["response_json_schema"]["required"][0] == "answer"`.
5. `bedrock_request_construction_no_network` — `BedrockClient::new_with_model("us.anthropic.claude-haiku-4-5-20251001-v1:0").with_region("eu-west-1")`; assert `api_endpoint() == "https://bedrock-runtime.eu-west-1.amazonaws.com"`, `get_api_key() == ""` (AWS creds, not API keys — bedrock.rs:203-206), and `format_request_body(&[user_msg("hi")], None, None)["messages"][0]["content"] == "hi"`. State in a test comment that this is the deliberate Bedrock boundary: the AWS SDK transport is not wiremockable.

Parse tests (call `parse_response` directly on canned JSON — exact canned bodies below):

6. `openai_parse_happy_path` — input `{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"ANSWER"},"finish_reason":"stop"}]}` → `Ok("ANSWER")`.
7. `openai_parse_null_content_is_parse_error` — same body but `"content":null` → `Err(ModelClientError::ParseError(_))` (openai.rs:30 `content: Option<String>`; assert with `matches!`).
8. `groq_parse_happy_path` — same happy body shape as OpenAI → `Ok("ANSWER")`.
9. `groq_parse_empty_choices_is_parse_error` — `{"id":"x","model":"m","choices":[]}` → `Err(ModelClientError::ParseError(_))`.
10. `anthropic_parse_text_happy_path` — `{"id":"msg_1","model":"claude-haiku-4-5","content":[{"type":"text","text":"ANSWER"}]}` → `Ok("ANSWER")`.
11. `anthropic_parse_tool_use_returns_input_json` — `{"id":"msg_2","model":"claude-haiku-4-5","content":[{"type":"tool_use","id":"tu_1","name":"response","input":{"answer":"42"}}]}` → `Ok` whose value parses (via `serde_json::from_str::<Value>`) to `{"answer":"42"}` (anthropic.rs:183-191 returns the tool input as JSON).
12. `anthropic_parse_empty_content_is_parse_error` — `{"id":"msg_3","model":"m","content":[]}` → `Err(ModelClientError::ParseError(_))`.
13. `gemini_parse_happy_path` — `{"candidates":[{"content":{"parts":[{"text":"ANSWER"}],"role":"model"}}]}` → `Ok("ANSWER")`.
14. `gemini_parse_empty_candidates_is_parse_error` — `{"candidates":[]}` → `Err(ModelClientError::ParseError(_))`. (Do not test a body missing the `candidates` key — that is a `Serialization` error, a different variant.)

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY cargo test --test provider_contract_tests` → `test result: ok.` with 14 passed.

### Step 4: Add wiremock end-to-end tests per HTTP provider

Append to the same file. Each test is `#[tokio::test]`, starts with `let _env = env_guard();`, starts a `MockServer`, sets that provider's `*_BASE_URL` to `mock_server.uri()`, and calls the real send path. Auth headers with an empty key are valid (the get_api_key default returns `""` when the env var is unset — mod.rs:231-239), so no API keys are needed. Do not assert on auth-key header values (a developer machine may have real keys in env); the only constant header worth asserting is Anthropic's `anthropic-version`.

Pattern (write this one, then adapt):

```rust
#[tokio::test]
async fn openai_mock_happy_path_end_to_end() {
    let _env = env_guard();
    let server = MockServer::start().await;
    std::env::set_var("OPENAI_BASE_URL", server.uri());

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": "my_schema", "strict": true }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"ANSWER"},"finish_reason":"stop"}]}"#,
        ))
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAIClient::new_with_model("gpt-4o-mini");
    let result = client
        .send_request_structured(http_client(), &[user_msg("hi")], Some(SCHEMA), Some("my_schema"))
        .await;
    assert_eq!(result.unwrap(), "ANSWER");

    std::env::remove_var("OPENAI_BASE_URL");
}
```

(`server.verify()` runs implicitly on `MockServer` drop, enforcing `.expect(1)` — an unmatched request-shape means the test fails, which doubles as the over-the-wire request assertion.)

Tests to add:

15. `openai_mock_happy_path_end_to_end` — as above.
16. `openai_mock_http_500_maps_to_http_error` — mock `POST /v1/chat/completions` → `ResponseTemplate::new(500).set_body_string("boom")`; call `send_request_structured` (no schema); assert `matches!(result, Err(ModelClientError::Http(500, ref body)) if body == "boom")`. This exercises mod.rs:178-182.
17. `anthropic_mock_happy_path_sends_version_header` — set `ANTHROPIC_BASE_URL`; mock `POST /v1/messages` with `.and(header("anthropic-version", "2023-06-01"))` and `.and(body_partial_json(json!({"max_tokens": 4096, "tool_choice": {"type": "tool", "name": "my_schema"}})))`; respond with the tool_use canned body from test 11; call `send_request_structured` with `SCHEMA` and `Some("my_schema")`; assert the `Ok` value parses to `{"answer":"42"}`. Remove the var at the end.
18. `anthropic_mock_http_429_maps_to_http_error` — mock → `ResponseTemplate::new(429).set_body_string("rate limited")`; assert `matches!(result, Err(ModelClientError::Http(429, _)))`. This exercises Anthropic's own override at anthropic.rs:230-234 (it does not inherit the mod.rs default).
19. `gemini_mock_happy_path_end_to_end` — set `GEMINI_BASE_URL` (from Step 2); mock `POST` with `path("/v1beta/models/gemini-2.5-flash:generateContent")` and `.and(body_partial_json(json!({"generationConfig": {"response_mime_type": "application/json"}})))`; respond with the test-13 happy body; use `GeminiClient::new_with_model("gemini-2.5-flash")` and `send_request_structured(..., Some(SCHEMA), None)`; assert `Ok("ANSWER")`. Remove the var.
20. `gemini_mock_http_500_maps_to_http_error` — 500 → `matches!(..., Err(ModelClientError::Http(500, _)))`.
21. `groq_mock_happy_path_end_to_end` — set `GROQ_BASE_URL`; mock `path("/openai/v1/chat/completions")` with `body_partial_json` on `response_format.json_schema`; respond with the test-8 happy body; assert `Ok("ANSWER")`. Remove the var.
22. `groq_mock_http_500_maps_to_http_error` — 500 → `Http(500, _)`.

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` → `test result: ok.` with 22 passed.

### Step 5: Add the batch positional-alignment test

Append test 23, `batch_alignment_mixed_success_failure`, going through the public wrapper `fetch_data_generic` (mod.rs:433-439; `run_batch` itself is private — do not expose it). Route by prompt content using `body_string_contains`:

```rust
#[tokio::test]
async fn batch_alignment_mixed_success_failure() {
    let _env = env_guard();
    let server = MockServer::start().await;
    std::env::set_var("OPENAI_BASE_URL", server.uri());

    let happy = |answer: &str| {
        ResponseTemplate::new(200).set_body_string(format!(
            r#"{{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{{"index":0,"message":{{"role":"assistant","content":"{answer}"}},"finish_reason":"stop"}}]}}"#
        ))
    };

    Mock::given(method("POST")).and(body_string_contains("ROW_ALPHA"))
        .respond_with(happy("ANSWER_ALPHA")).expect(1).mount(&server).await;
    Mock::given(method("POST")).and(body_string_contains("ROW_BETA"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(1).mount(&server).await;
    Mock::given(method("POST")).and(body_string_contains("ROW_GAMMA"))
        .respond_with(happy("ANSWER_GAMMA")).expect(1).mount(&server).await;

    let client = create_client(Provider::OpenAI, "gpt-4o-mini");
    let messages = vec![
        "ROW_ALPHA".to_string(),
        "ROW_BETA".to_string(),
        "ROW_GAMMA".to_string(),
    ];
    let results = fetch_data_generic(&*client, &messages).await;

    // Row N's failure yields None at index N without shifting neighbors.
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].as_deref(), Some("ANSWER_ALPHA"));
    assert_eq!(results[1], None);
    assert_eq!(results[2].as_deref(), Some("ANSWER_GAMMA"));

    std::env::remove_var("OPENAI_BASE_URL");
}
```

Note: the failing row prints `Error fetching from openai: HTTP Error 500: boom` to stderr (mod.rs:387) — expected noise, not a failure.

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` → `test result: ok.` with 23 passed.

### Step 6: Run the full repo gates

```
cargo fmt --all
cargo fmt --all -- --check
RUSTFLAGS="-Dwarnings" cargo clippy --all-features
cargo test --lib
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY cargo test
```

**Verify**: every command exits 0. The last one runs all test binaries; `model_client_tests` prints skip messages without keys — that is a pass, not a failure.

## Test plan

All new tests live in `tests/provider_contract_tests.rs` (created here; the structural exemplar for locking/skipping conventions is `tests/model_client_tests.rs`). Coverage matrix — each of the 4 HTTP providers gets at least request-shape + parse-happy + parse-error, plus HTTP-error mapping and one wiremock happy path:

- OpenAI: tests 1, 6, 7, 15, 16, 23
- Groq: tests 2, 8, 9, 21, 22
- Anthropic: tests 3, 10, 11, 12, 17, 18
- Gemini: tests 4, 13, 14, 19, 20
- Bedrock (no-network boundary, stated in-file): test 5
- Batch alignment through `fetch_data_generic`: test 23

Verification: the keyless command in "Commands you will need" → 23 passed, 0 failed.

## Done criteria

Machine-checkable. ALL must hold (run from the repo root):

- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` exits 0 with `23 passed; 0 failed`
- [ ] `cargo test --lib` exits 0 (no regression in unit tests)
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY cargo test` exits 0 (whole suite keyless)
- [ ] `grep -c "GEMINI_BASE_URL" src/model_client/gemini.rs` prints `1` and `grep -c "GROQ_BASE_URL" src/model_client/groq.rs` prints `1`
- [ ] `grep -n "wiremock" Cargo.toml` shows the entry under `[dev-dependencies]` only (not `[dependencies]`)
- [ ] `git status --porcelain` shows changes only to `Cargo.toml`, `Cargo.lock`, `src/model_client/gemini.rs`, `src/model_client/groq.rs`, `tests/provider_contract_tests.rs` (and `plans/README.md` if it exists)
- [ ] `plans/README.md` status row updated, if that file exists

## STOP conditions

Stop and report back (do not improvise) if:

- The excerpts in "Current state" don't match the live code (e.g. `gemini.rs` `api_endpoint` is no longer at lines 62–64 with the hardcoded URL, or `openai.rs`/`anthropic.rs` no longer read `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL`) — the codebase has drifted.
- Adding the `GEMINI_BASE_URL` or `GROQ_BASE_URL` override turns out to require touching `apply_auth`, `get_api_key`, or any auth/header logic — the finding explicitly forbids restructuring an auth flow for mockability.
- `wiremock = "0.6"` fails to resolve or compile against this repo's tokio/reqwest versions after one honest attempt (do not downgrade tokio/reqwest or add feature flags to fix it).
- `cargo test --test provider_contract_tests` fails twice after a reasonable fix attempt, or passes only when tests run single-threaded (`--test-threads=1`) — that means the ENV_LOCK pattern is not actually serializing env access; report rather than hard-coding `--test-threads=1`.
- Making any assertion pass appears to require changing request/response logic in `src/model_client/mod.rs`, `openai.rs`, `anthropic.rs`, or `bedrock.rs` — a shape mismatch there is a finding to report, not something to "fix" silently.
- `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` fails on pre-existing code unrelated to your changes.

## Maintenance notes

- These tests pin the provider wire contracts. Any deliberate change to request framing (structured-output blocks, `max_tokens`, headers) or response parsing must update the matching contract test in the same PR — that is the point of the suite.
- plans/008 (retry logic) will change the send path; the wiremock tests here are the natural place to add retry assertions (e.g. `.expect(2)` after one 429) — keep the ENV_LOCK pattern when doing so.
- Reviewer should scrutinize: (a) wiremock is under `[dev-dependencies]` only; (b) every test that sets a `*_BASE_URL` var holds `env_guard()` for its whole body and removes the var before returning; (c) the two `api_endpoint` diffs are byte-for-byte in the style of openai.rs:57-61 with no auth changes; (d) no `dotenvy` usage in the new file.
- Deliberately deferred: wiremocking the OpenAI embeddings client (`openai.rs:190-192` hardcodes the embeddings endpoint — adding an override there was out of the finding's scope); Bedrock transport-level testing (needs `aws-smithy-mocks` or an SDK interceptor — a separate decision, not smuggled in here); Anthropic `anthropic-beta` cache-header assertions (belongs with cache-focused work).
- CI enforcement of these tests arrives with plans/002-ci-verification-gates.md; until it lands, run the keyless command locally before merging changes to `src/model_client/`.
