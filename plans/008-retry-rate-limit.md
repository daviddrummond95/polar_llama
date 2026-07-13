# Plan 008: Retry 429/5xx with bounded exponential backoff honoring Retry-After

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/model_client/mod.rs src/model_client/anthropic.rs src/model_client/openai.rs tests/ CHANGELOG.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/003-provider-http-mock-tests.md (adds the `wiremock` dev-dependency and establishes the HTTP-mock testing pattern; this plan's mock tests cannot compile without it)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

polar-llama fans out up to 64 concurrent HTTP requests per batch (one per
DataFrame row). Under that concurrency, HTTP 429 (rate limit) is the most
common real-world failure — and today any non-2xx response is terminal: the
row permanently fails with `ModelClientError::Http(status, body)` and the user
gets a `null`/error cell even though the request would have succeeded moments
later. There is no retry/backoff logic anywhere in `src/model_client/`
(verified across mod.rs, openai.rs, anthropic.rs, gemini.rs, groq.rs,
bedrock.rs). After this plan lands, transient failures (429, 502, 503, 504,
and reqwest connect/timeout errors) are retried with bounded exponential
backoff plus full jitter, honoring the provider's `Retry-After` header, so
large batches complete instead of dropping rows.

## Current state

Relevant files:

- `src/model_client/mod.rs` — shared HTTP plumbing; the trait-default
  `send_request_structured` (lines 159–183) is the send path used by the
  OpenAI, Gemini, and Groq clients. Also holds the env-knob exemplar
  `max_concurrency()` (lines 35–41).
- `src/model_client/anthropic.rs` — **overrides** `send_request_structured`
  (lines 202–235) to inject `anthropic-beta` headers; it duplicates the
  send/status/text logic, so the retry helper must be applied here too or
  Anthropic requests get no retries. (The original finding's scope list
  missed this file.)
- `src/model_client/openai.rs` — `generate_embeddings` (lines 202–237) is a
  second independent send path (embeddings). `embedding_endpoint` (lines
  190–192) is hardcoded to `https://api.openai.com` and does NOT honor
  `OPENAI_BASE_URL` (unlike `api_endpoint`, lines 57–61), which blocks
  mock-testing the embeddings path.
- `src/model_client/bedrock.rs` — sends via the AWS SDK, not reqwest:
  `send_request` (lines 155–187) calls `aws-sdk-bedrockruntime`'s
  `converse()`, and its `send_request_structured` override (lines 189–201)
  just delegates to `send_request`. The AWS SDK ships its own configurable
  retry strategy (standard/adaptive retries via `aws-config` with
  `behavior-version-latest`). **Do not touch it** — wrapping SDK calls in
  our HTTP retry loop would double-retry.
- `src/model_client/gemini.rs`, `src/model_client/groq.rs` — no send
  overrides; they inherit the trait default, so they get retries for free
  once mod.rs is fixed. No edits needed there.

### Excerpt: trait-default send path, `src/model_client/mod.rs:158-183`

```rust
    /// Send a request with structured output support
    async fn send_request_structured(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>
    ) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = self.format_request_body(messages, schema, model_name);

        let response = self
            .apply_auth(client.post(self.api_endpoint()), &api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }
```

### Excerpt: the env-knob idiom to copy, `src/model_client/mod.rs:33-41`

```rust
/// Maximum number of concurrent in-flight requests per batch.
/// Override with the POLAR_LLAMA_MAX_CONCURRENCY environment variable.
fn max_concurrency() -> usize {
    std::env::var("POLAR_LLAMA_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64)
}
```

(Note: for the new `max_retries()` knob you must NOT copy the
`.filter(|v| *v > 0)` line — `0` is a meaningful value that disables retries.)

### Excerpt: Anthropic override tail, `src/model_client/anthropic.rs:227-234`

```rust
        let response = request.json(&body).send().await?;
        let status = response.status();
        let text = response.text().await?;
        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
```

(`request` at that point is `self.apply_auth(client.post(self.api_endpoint()), &api_key)`
plus an optional `anthropic-beta` header added at lines 214–225.)

### Excerpt: embeddings send path, `src/model_client/openai.rs:215-237`

```rust
        let response = client
            .post(self.embedding_endpoint())
            .bearer_auth(api_key)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            let embedding_response: OpenAIEmbeddingResponse =
                serde_json::from_str(&text)?;

            // Sort by index to ensure correct order
            let mut sorted_data = embedding_response.data;
            sorted_data.sort_by_key(|d| d.index);

            Ok(sorted_data.into_iter().map(|d| d.embedding).collect())
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
```

### Excerpt: hardcoded embedding endpoint, `src/model_client/openai.rs:190-192`

```rust
    fn embedding_endpoint(&self) -> String {
        "https://api.openai.com/v1/embeddings".to_string()
    }
```

### Conventions and constraints that apply

- Errors use `ModelClientError` (`src/model_client/mod.rs:92-123`); reqwest
  errors convert via the existing `From<reqwest::Error>` impl. Match this.
- `Cargo.toml` already enables tokio's `time` feature
  (`tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time"] }`),
  so `tokio::time::sleep` is available. Do not add tokio features.
- **Do NOT add a `rand` dependency.** rand 0.8/0.9 currently trips a RUSTSEC
  unsound advisory in this repo's `cargo audit` CI job. Use the
  hash-of-clock-nanos jitter given in Step 1.
- CI compiles with `RUSTFLAGS=-Dwarnings`, so any warning is a build failure.
- `pub mod model_client` is exported from `src/lib.rs:4`, and
  `pub mod openai` from `src/model_client/mod.rs:1` — integration tests can
  reach `polar_llama::model_client::{ModelClient, EmbeddingClient, Message, ModelClientError}`
  and `polar_llama::model_client::openai::{OpenAIClient, OpenAIEmbeddingClient}`.
  See `tests/model_client_tests.rs:1-4` for the import style.
- `tests/model_client_tests.rs` needs live API keys (it self-skips without
  them). Your new test file must need NO real keys — dummy values only.
- Changelog: `CHANGELOG.md` follows Keep-a-Changelog; add under
  `## [Unreleased]`. Exemplar entry style (line 47):
  `- \`POLAR_LLAMA_MAX_CONCURRENCY\` environment variable to bound concurrent in-flight requests per batch (default 64; previously unbounded)`.

### Retry policy (the exact behavior to implement)

- Retry ONLY on: HTTP 429, 502, 503, 504, and reqwest transport errors where
  `e.is_connect()` or `e.is_timeout()` is true. Everything else (400, 401,
  403, 404, 500, parse errors, …) fails fast with no retry.
- `POLAR_LLAMA_MAX_RETRIES` = number of **retries after the initial
  attempt**. Default `2` (so up to 3 total attempts). `0` disables retries
  (exactly 1 attempt). Unparseable values fall back to the default.
- Delay before retry N (N starting at 0): if the response carries a
  `Retry-After` header in integer-seconds form, sleep that many seconds
  capped at 30s; otherwise full-jitter exponential backoff — a uniform-ish
  random duration in `[0, min(30s, 500ms * 2^N))`.
- When retries are exhausted, surface the LAST response as
  `ModelClientError::Http(status, body)` exactly as today (body preserved).

## Commands you will need

All from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Precondition check | `grep -n "wiremock" Cargo.toml` | at least one match in `[dev-dependencies]` (else STOP — plan 003 not landed) |
| Format check | `cargo fmt --all -- --check` | exit 0, no output |
| Format fix | `cargo fmt --all` | exit 0 |
| Lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0, no warnings/errors |
| Rust unit tests | `cargo test --lib` | exit 0, all pass (includes the new retry unit tests) |
| New mock tests | `cargo test --test retry_mock_tests` | exit 0, `6 passed` |
| Compile check | `cargo check --all-features` | exit 0 |

Notes: `cargo test --lib` and `--test retry_mock_tests` need NO API keys.
Never run `cargo test --test model_client_tests` as a gate — it needs live
keys and self-skips without them.

## Scope

**In scope** (the only files you may modify/create):

- `src/model_client/mod.rs` — add the retry helper + env knob; rewire the
  trait-default `send_request_structured`; add unit tests.
- `src/model_client/anthropic.rs` — rewire its `send_request_structured`
  override through the shared helper.
- `src/model_client/openai.rs` — rewire `generate_embeddings` through the
  helper; make `embedding_endpoint` honor `OPENAI_BASE_URL`.
- `tests/retry_mock_tests.rs` — create (wiremock-based retry tests).
- `CHANGELOG.md` — one `[Unreleased]` entry.
- `plans/README.md` — status row only (if it exists and no reviewer told you otherwise).

**Out of scope** (do NOT touch, even though they look related):

- `src/model_client/bedrock.rs` — goes through the AWS SDK, which has its own
  retry configuration (adaptive retries via `aws-config`); adding our HTTP
  retry loop would double-retry. Leave it alone.
- `src/model_client/gemini.rs`, `src/model_client/groq.rs` — they inherit the
  trait default and get retries automatically; no edits.
- `Cargo.toml` — the `wiremock` dev-dependency comes from plan 003. If it is
  missing, STOP; do not add it yourself.
- A token-bucket/global rate limiter — future work, see Maintenance notes.
- The Python layer (`polar_llama/`), `src/cost.rs`, `src/expressions.rs`.
- Any change to public API signatures or error variants.

## Git workflow

- Branch: `advisor/008-retry-rate-limit` (branch off `main`).
- Commit per step or per logical unit. Message style: sentence-case
  imperative summary, matching the repo log — e.g.
  `Add bounded retry with backoff for transient provider errors`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Preconditions

1. Run the drift check from the header block.
2. `grep -n "wiremock" Cargo.toml` — must match in `[dev-dependencies]`.
   If it does not, plan 003 has not landed: STOP and report.
3. `grep -n "OPENAI_BASE_URL" src/model_client/openai.rs` — defensive check
   only: plan 003 explicitly does NOT modify `openai.rs` (it is in that
   plan's out-of-scope list), so expect exactly one match today, inside
   `api_endpoint` (line 58). If `embedding_endpoint` unexpectedly already
   honors `OPENAI_BASE_URL`, skip the endpoint part of Step 3.

**Verify**: both greps ran; wiremock present → proceed.

### Step 1: Add the retry helper to mod.rs and rewire the trait default

(Do these together in one step — a helper with no caller is dead code and
fails the `-Dwarnings` build.)

In `src/model_client/mod.rs`, directly below `max_concurrency()` (after line
41), add:

```rust
/// Maximum number of retry attempts (after the initial request) for
/// transient failures: HTTP 429/502/503/504 and reqwest connect/timeout
/// errors. Override with the POLAR_LLAMA_MAX_RETRIES environment variable;
/// 0 disables retries.
fn max_retries() -> u32 {
    std::env::var("POLAR_LLAMA_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(2)
}

/// Base delay for exponential backoff between retries.
const RETRY_BASE_DELAY_MS: u64 = 500;
/// Upper bound on any single retry sleep (computed backoff or Retry-After).
const RETRY_MAX_DELAY_MS: u64 = 30_000;

/// Statuses worth retrying: rate limit and transient upstream failures.
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 502 | 503 | 504)
}

/// Pseudo-random fraction in [0, 1) without a rand dependency: SipHash
/// (DefaultHasher) over a process-wide counter and the current clock nanos.
/// rand 0.8/0.9 currently trips a RUSTSEC unsound advisory in cargo-audit,
/// so we deliberately avoid adding it for jitter.
fn jitter_fraction() -> f64 {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    COUNTER.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
        .hash(&mut hasher);
    (hasher.finish() % 1_000_000) as f64 / 1_000_000.0
}

/// Full-jitter exponential backoff: uniform-ish in
/// [0, min(RETRY_MAX_DELAY_MS, RETRY_BASE_DELAY_MS * 2^attempt)).
fn backoff_delay(attempt: u32) -> Duration {
    let exp = RETRY_BASE_DELAY_MS.saturating_mul(2u64.saturating_pow(attempt));
    let cap_ms = exp.min(RETRY_MAX_DELAY_MS);
    Duration::from_millis((cap_ms as f64 * jitter_fraction()) as u64)
}

/// Parse a Retry-After header in its integer-seconds form, capped at
/// RETRY_MAX_DELAY_MS. The HTTP-date form is not parsed (falls back to
/// computed backoff).
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let secs = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(Duration::from_millis(secs.saturating_mul(1000).min(RETRY_MAX_DELAY_MS)))
}

/// Send an HTTP request with bounded retries on transient failures.
///
/// `build_request` is invoked once per attempt (a reqwest request cannot be
/// re-sent). Retries on 429/502/503/504 and on reqwest connect/timeout
/// errors, honoring Retry-After (integer-seconds form) when present, else
/// full-jitter exponential backoff. Returns the final status and body text;
/// the caller maps non-success statuses to errors, so exhausted retries
/// still surface the last response body.
pub(crate) async fn send_with_retry<F>(
    build_request: F,
) -> Result<(reqwest::StatusCode, String), ModelClientError>
where
    F: Fn() -> reqwest::RequestBuilder + Send + Sync,
{
    let max_retries = max_retries();
    let mut attempt: u32 = 0;
    loop {
        match build_request().send().await {
            Ok(response) => {
                let status = response.status();
                if !is_retryable_status(status) || attempt >= max_retries {
                    let text = response.text().await?;
                    return Ok((status, text));
                }
                let delay = parse_retry_after(response.headers())
                    .unwrap_or_else(|| backoff_delay(attempt));
                tokio::time::sleep(delay).await;
            }
            Err(e) if (e.is_connect() || e.is_timeout()) && attempt < max_retries => {
                tokio::time::sleep(backoff_delay(attempt)).await;
            }
            Err(e) => return Err(e.into()),
        }
        attempt += 1;
    }
}
```

(`Duration` is already imported at mod.rs:11: `use std::time::Duration;`.)

Then replace the body of the trait-default `send_request_structured`
(currently mod.rs:159–183, shown in Current state) with:

```rust
        let api_key = self.get_api_key();
        let body = self.format_request_body(messages, schema, model_name);

        let (status, text) = send_with_retry(|| {
            self.apply_auth(client.post(self.api_endpoint()), &api_key)
                .json(&body)
        })
        .await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
```

Also add unit tests for the pure helpers at the bottom of mod.rs (these run
today with `cargo test --lib`, no wiremock needed):

```rust
#[cfg(test)]
mod retry_tests {
    use super::*;

    #[test]
    fn retryable_status_matrix() {
        for code in [429u16, 502, 503, 504] {
            assert!(is_retryable_status(reqwest::StatusCode::from_u16(code).unwrap()));
        }
        for code in [200u16, 400, 401, 403, 404, 500] {
            assert!(!is_retryable_status(reqwest::StatusCode::from_u16(code).unwrap()));
        }
    }

    #[test]
    fn backoff_stays_within_bounds() {
        for attempt in 0..10 {
            let cap = RETRY_BASE_DELAY_MS
                .saturating_mul(2u64.saturating_pow(attempt))
                .min(RETRY_MAX_DELAY_MS);
            let d = backoff_delay(attempt);
            assert!(d.as_millis() as u64 <= cap, "attempt {attempt}: {d:?} > {cap}ms");
        }
    }

    #[test]
    fn retry_after_seconds_form_parsed_and_capped() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "2".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(2)));

        headers.insert(reqwest::header::RETRY_AFTER, "9999".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_millis(RETRY_MAX_DELAY_MS)));

        // HTTP-date form is unsupported -> None (falls back to backoff)
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), None);

        assert_eq!(parse_retry_after(&reqwest::header::HeaderMap::new()), None);
    }

    #[test]
    fn max_retries_env_parsing() {
        // NOTE: env vars are process-global; this is the only lib unit test
        // touching POLAR_LLAMA_MAX_RETRIES, and it restores state at the end.
        std::env::remove_var("POLAR_LLAMA_MAX_RETRIES");
        assert_eq!(max_retries(), 2);
        std::env::set_var("POLAR_LLAMA_MAX_RETRIES", "0");
        assert_eq!(max_retries(), 0);
        std::env::set_var("POLAR_LLAMA_MAX_RETRIES", "5");
        assert_eq!(max_retries(), 5);
        std::env::set_var("POLAR_LLAMA_MAX_RETRIES", "not-a-number");
        assert_eq!(max_retries(), 2);
        std::env::remove_var("POLAR_LLAMA_MAX_RETRIES");
    }
}
```

**Verify**: `cargo check --all-features` → exit 0; then `cargo test --lib` →
exit 0, the 4 new `retry_tests` pass.

### Step 2: Rewire the Anthropic override through the helper

In `src/model_client/anthropic.rs`, in its `send_request_structured` override
(lines 202–235), keep the api-key/body/beta-header computation but move the
request construction into the retry closure. Replace lines 212–234 — from
`let mut request = ...` through the closing `}` of the
`if status.is_success() { ... } else { ... }` block; the method's own closing
brace at line 235 stays — with:

```rust
        // Prompt-caching beta header(s); extended (1h) TTL needs the extra beta.
        let beta_header: Option<&'static str> = if messages.iter().any(|msg| msg.cache_control.is_some()) {
            let needs_extended_ttl = messages.iter().any(|msg| {
                msg.cache_control.as_ref().and_then(|cc| cc.ttl.as_deref()) == Some("1h")
            });
            Some(if needs_extended_ttl {
                "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"
            } else {
                "prompt-caching-2024-07-31"
            })
        } else {
            None
        };

        let (status, text) = super::send_with_retry(|| {
            let mut request = self.apply_auth(client.post(self.api_endpoint()), &api_key);
            if let Some(beta) = beta_header {
                request = request.header("anthropic-beta", beta);
            }
            request.json(&body)
        })
        .await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
```

The `let api_key = self.get_api_key();` and
`let body = self.format_request_body(messages, schema, model_name);` lines at
209–210 stay as they are.

**Verify**: `cargo check --all-features` → exit 0.

### Step 3: Rewire the embeddings path and un-hardcode its endpoint

In `src/model_client/openai.rs`:

1. Unless plan 003 already did it (Step 0 check), replace `embedding_endpoint`
   (lines 190–192) with the same base-URL pattern as `api_endpoint`
   (lines 57–61):

```rust
    fn embedding_endpoint(&self) -> String {
        let base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        format!("{}/v1/embeddings", base.trim_end_matches('/'))
    }
```

2. In `generate_embeddings` (lines 202–237), replace the send/status/text
   block (lines 215–223, shown in Current state) with:

```rust
        let (status, text) = super::send_with_retry(|| {
            client
                .post(self.embedding_endpoint())
                .bearer_auth(&api_key)
                .json(&request_body)
        })
        .await?;
```

   (Note `.bearer_auth(&api_key)` — borrow, not move, since the closure runs
   once per attempt.) The success/error handling below it
   (`if status.is_success() { ... } else { Err(ModelClientError::Http(...)) }`)
   stays unchanged.

**Verify**: `cargo fmt --all` then
`RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0, no warnings.

### Step 4: Write the wiremock retry tests

Create `tests/retry_mock_tests.rs` with exactly these six tests. They need no
real API keys — `OPENAI_API_KEY` is set to the dummy literal `test-key`.
Because env vars are process-global and cargo runs tests in this binary in
parallel threads, EVERY test takes the shared `ENV_LOCK` first (same idiom as
`NET_TEST_LOCK` + `network_guard()` in `tests/model_client_tests.rs:16-20`).

```rust
//! Retry/backoff behavior tests against a wiremock server. No live API keys
//! required; OPENAI_BASE_URL is pointed at the local mock.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use polar_llama::model_client::openai::{OpenAIClient, OpenAIEmbeddingClient};
use polar_llama::model_client::{EmbeddingClient, Message, ModelClient, ModelClientError};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Serializes tests: they all mutate process-global env vars
/// (OPENAI_BASE_URL, POLAR_LLAMA_MAX_RETRIES).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn set_env(server: &MockServer) {
    std::env::set_var("OPENAI_BASE_URL", server.uri());
    std::env::set_var("OPENAI_API_KEY", "test-key");
    std::env::remove_var("POLAR_LLAMA_MAX_RETRIES");
}

fn clear_env() {
    std::env::remove_var("OPENAI_BASE_URL");
    std::env::remove_var("POLAR_LLAMA_MAX_RETRIES");
}

const CHAT_OK: &str = r#"{"id":"cmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
const EMBED_OK: &str = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"text-embedding-3-small"}"#;

fn user_message() -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: "hi".to_string(),
        cache_control: None,
    }]
}

async fn requests_received(server: &MockServer) -> usize {
    server.received_requests().await.map(|r| r.len()).unwrap_or(0)
}

#[tokio::test]
async fn retries_429_then_succeeds() {
    let _guard = env_guard();
    let server = MockServer::start().await;

    // First call: 429. Subsequent calls: 200. wiremock serves mounts in
    // registration order; `up_to_n_times(1)` exhausts the first mock.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(CHAT_OK, "application/json"))
        .mount(&server)
        .await;

    set_env(&server);
    let client = OpenAIClient::new_with_model("gpt-4o-mini");
    let http = reqwest::Client::new();
    let result = client.send_request(&http, &user_message()).await;
    clear_env();

    assert_eq!(result.unwrap(), "ok");
    assert_eq!(requests_received(&server).await, 2);
}

#[tokio::test]
async fn honors_retry_after_header() {
    let _guard = env_guard();
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "1"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(CHAT_OK, "application/json"))
        .mount(&server)
        .await;

    set_env(&server);
    let client = OpenAIClient::new_with_model("gpt-4o-mini");
    let http = reqwest::Client::new();
    let start = Instant::now();
    let result = client.send_request(&http, &user_message()).await;
    let elapsed = start.elapsed();
    clear_env();

    assert_eq!(result.unwrap(), "ok");
    assert!(
        elapsed >= Duration::from_secs(1),
        "expected >= 1s wait from Retry-After, got {elapsed:?}"
    );
}

#[tokio::test]
async fn persistent_429_fails_after_max_attempts() {
    let _guard = env_guard();
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    set_env(&server);
    let client = OpenAIClient::new_with_model("gpt-4o-mini");
    let http = reqwest::Client::new();
    let result = client.send_request(&http, &user_message()).await;
    clear_env();

    match result {
        Err(ModelClientError::Http(429, _)) => {}
        other => panic!("expected Http(429, _), got {other:?}"),
    }
    // Default POLAR_LLAMA_MAX_RETRIES=2 -> 3 total attempts.
    assert_eq!(requests_received(&server).await, 3);
}

#[tokio::test]
async fn does_not_retry_400() {
    let _guard = env_guard();
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .mount(&server)
        .await;

    set_env(&server);
    let client = OpenAIClient::new_with_model("gpt-4o-mini");
    let http = reqwest::Client::new();
    let result = client.send_request(&http, &user_message()).await;
    clear_env();

    match result {
        Err(ModelClientError::Http(400, body)) => assert_eq!(body, "bad request"),
        other => panic!("expected Http(400, _), got {other:?}"),
    }
    assert_eq!(requests_received(&server).await, 1);
}

#[tokio::test]
async fn max_retries_zero_disables_retries() {
    let _guard = env_guard();
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    set_env(&server);
    std::env::set_var("POLAR_LLAMA_MAX_RETRIES", "0");
    let client = OpenAIClient::new_with_model("gpt-4o-mini");
    let http = reqwest::Client::new();
    let result = client.send_request(&http, &user_message()).await;
    clear_env();

    match result {
        Err(ModelClientError::Http(429, _)) => {}
        other => panic!("expected Http(429, _), got {other:?}"),
    }
    assert_eq!(requests_received(&server).await, 1);
}

#[tokio::test]
async fn embeddings_retry_429_then_succeed() {
    let _guard = env_guard();
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(EMBED_OK, "application/json"))
        .mount(&server)
        .await;

    set_env(&server);
    let client = OpenAIEmbeddingClient::new_with_model("text-embedding-3-small");
    let http = reqwest::Client::new();
    let result = client
        .generate_embeddings(&http, &["hello".to_string()])
        .await;
    clear_env();

    let embeddings = result.unwrap();
    assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
    assert_eq!(requests_received(&server).await, 2);
}
```

If plan 003's wiremock version exposes a different API for the patterns used
here (`up_to_n_times`, `insert_header`, `set_body_raw`, `received_requests`
all exist in wiremock 0.6), adapt the call names only — do not change what
each test asserts.

**Verify**: `cargo test --test retry_mock_tests` → exit 0, `6 passed; 0 failed`.

### Step 5: Changelog entry

Add under `## [Unreleased]` in `CHANGELOG.md` (create an `### Added`
subsection there if absent):

```markdown
### Added
- Bounded retry with exponential backoff + full jitter for transient provider failures (HTTP 429/502/503/504 and connect/timeout errors), honoring `Retry-After`. Configurable via `POLAR_LLAMA_MAX_RETRIES` (retries after the first attempt; default 2, `0` disables). Applies to all HTTP providers including the OpenAI embeddings path; Bedrock keeps the AWS SDK's own retry handling. Note: retries multiply provider spend for the retried request; cost estimates do not model retried attempts.
```

**Verify**: `git diff CHANGELOG.md` shows only the new entry under `[Unreleased]`.

### Step 6: Full gate

Run, in order, from the repo root:

1. `cargo fmt --all -- --check` → exit 0
2. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0
3. `cargo test --lib` → exit 0
4. `cargo test --test retry_mock_tests` → exit 0, 6 passed
5. `git status --short` → only the in-scope files modified/created

**Verify**: all five results as stated.

## Test plan

- New unit tests in `src/model_client/mod.rs` (`mod retry_tests`, Step 1):
  retryable-status matrix, backoff bounds, Retry-After parsing (seconds, cap,
  HTTP-date fallback, absent), `POLAR_LLAMA_MAX_RETRIES` parsing
  (default/0/5/garbage).
- New integration tests in `tests/retry_mock_tests.rs` (Step 4), modeled
  structurally on `tests/model_client_tests.rs` (static-Mutex guard idiom)
  but using wiremock instead of live APIs:
  1. 429-then-200 → success after retry (2 requests observed).
  2. Retry-After: 1s honored → elapsed >= 1s.
  3. Persistent 429 → `Http(429, _)` after exactly 3 attempts (default).
  4. 400 → fail fast, exactly 1 attempt, body preserved.
  5. `POLAR_LLAMA_MAX_RETRIES=0` → exactly 1 attempt.
  6. Embeddings path: 429-then-200 → success (covers the second send path).
- Verification: `cargo test --lib` and `cargo test --test retry_mock_tests`
  → all pass, no API keys set.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `cargo test --lib` exits 0 (includes 4 new `retry_tests`)
- [ ] `cargo test --test retry_mock_tests` exits 0 with `6 passed`
- [ ] `grep -n "send_with_retry" src/model_client/mod.rs src/model_client/anthropic.rs src/model_client/openai.rs` → at least one match in each of the three files
- [ ] `grep -n "POLAR_LLAMA_MAX_RETRIES" src/model_client/mod.rs CHANGELOG.md` → matches in both
- [ ] `grep -n "send_with_retry\|POLAR_LLAMA_MAX_RETRIES" src/model_client/bedrock.rs src/model_client/gemini.rs src/model_client/groq.rs` → no matches (those files untouched)
- [ ] `git status --short` shows changes only to: `src/model_client/mod.rs`, `src/model_client/anthropic.rs`, `src/model_client/openai.rs`, `tests/retry_mock_tests.rs`, `CHANGELOG.md` (and `plans/README.md` if you updated the index)
- [ ] `plans/README.md` status row updated (if the index exists and no reviewer said they maintain it)

## STOP conditions

Stop and report back (do not improvise) if:

- `wiremock` is not in `Cargo.toml` `[dev-dependencies]` (plan 003 has not
  landed — this plan depends on it; do not add the dependency yourself).
- The code at mod.rs:158-183, anthropic.rs:202-235, or openai.rs:202-237
  does not match the "Current state" excerpts (drift since `afc78da`).
- The `send_with_retry` closure fails to compile due to `Send`/`Sync`/lifetime
  bounds in the `async_trait` context and one bound adjustment (e.g. dropping
  `Sync`, or adding `+ '_`) doesn't fix it — do not switch to cloning
  requests or restructuring the trait.
- Reading `Retry-After` turns out to be impossible before the body is
  consumed in some path you encounter (it should not be — headers are
  available on `reqwest::Response` before `.text()`; the helper reads them
  first by design).
- Making the tests pass appears to require modifying `bedrock.rs`,
  `gemini.rs`, `groq.rs`, `Cargo.toml`, or any Python file.
- A test fails twice in a row after a reasonable fix attempt — especially
  the timing assertion in `honors_retry_after_header` (possible CI-load
  flake; report rather than weakening the assertion).

## Maintenance notes

For whoever owns this code after the change lands:

- **Cost interaction**: retries multiply provider spend for the affected
  request (a retried generation may bill twice on some 5xx failures).
  `src/cost.rs` estimates tokens per logical request and does not model
  retried attempts; if precise spend accounting is ever needed, retry counts
  must be surfaced. The changelog entry documents this.
- **Trait override hazard**: any future provider that overrides
  `send_request_structured` (as Anthropic does) bypasses the trait-default
  retry loop and MUST route its send through `send_with_retry`. A reviewer
  should check exactly this in any new-provider PR.
- **Bedrock deliberately excluded**: it uses the AWS SDK's own retry
  strategy; wrapping it here would double-retry.
- **Deferred follow-ups** (explicitly out of scope here):
  - A token-bucket / global rate limiter that paces requests *before* hitting
    429s (the retry loop is damage control, not pacing).
  - Parsing the HTTP-date form of `Retry-After` (currently falls back to
    computed backoff).
  - Retrying HTTP 500 or `Retry-After` on 503-with-date; revisit only with
    provider evidence.
- **Reviewer focus**: the borrow-per-attempt closure pattern (nothing moved
  into the closure), the `attempt >= max_retries` boundary (default 2 retries
  = 3 total attempts, `0` = 1 attempt), and that terminal errors still carry
  the response body exactly as before.
