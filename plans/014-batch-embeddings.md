# Plan 014: Send embeddings in provider-batch chunks instead of one HTTP call per row

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index (if `plans/README.md` does not exist, skip this).
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/model_client/mod.rs src/model_client/openai.rs tests/embedding_batch_tests.rs`
>
> **Expected drift** — this plan depends on two other plans that touch these
> paths, so some drift is normal:
> - `src/model_client/mod.rs` SHOULD show changes from
>   plans/005-embeddings-fail-loudly.md (a `ModelClientError::Unsupported`
>   variant near the top of the enum, a `Result`-returning
>   `create_embedding_client`, and a `#[cfg(test)] mod tests` at the bottom).
>   `embed_one` and `fetch_embeddings_generic` themselves are NOT changed by
>   plan 005 — if their bodies differ from the excerpts below, that is real
>   drift: STOP.
> - `src/model_client/openai.rs` should show NO drift (plans 003 and 005 both
>   exclude it). Any change there is real drift: STOP.
> - `tests/embedding_batch_tests.rs` should not exist yet (this plan creates
>   it). If it exists, someone already started this work: STOP.
>
> Because plan 005 shifts line numbers in `src/model_client/mod.rs` by a few
> lines, all `mod.rs` line numbers below are as of commit `afc78da`. Locate
> code by symbol name (`embed_one`, `fetch_embeddings_generic`,
> `max_concurrency`), not by line number.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/005-embeddings-fail-loudly.md, plans/003-provider-http-mock-tests.md
- **Category**: perf
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

`fetch_embeddings_generic` sends one HTTP request per input row: it maps every
text to `embed_one`, which wraps the text in a 1-element slice and calls
`generate_embeddings`. But `generate_embeddings` already accepts `&[String]`
and serializes `"input": texts` — OpenAI's `/v1/embeddings` endpoint accepts up
to 2048 inputs per call. Embedding a 10,000-row column therefore costs 10,000
round-trips (merely concurrency-capped at 64) when ~20 batched requests would
do. This plan chunks texts into batches (default 512, env-overridable), sends
one request per chunk, and preserves the existing per-row failure isolation by
falling back to per-item requests for a chunk only when that chunk's batched
request fails ("retry-shrink"). Result: order-of-magnitude fewer HTTP requests
for large columns, with identical output semantics.

## Current state

### Files and roles

- `src/model_client/mod.rs` — `EmbeddingClient` trait (line 244 at `afc78da`);
  `max_concurrency()` env-override idiom (lines 33–41); the per-row functions
  `embed_one` (lines 488–499) and `fetch_embeddings_generic` (lines 501–512)
  this plan reworks. After plan 005 lands, this file also has a
  `#[cfg(test)] mod tests` at the bottom (added by 005) — this plan appends
  one unit test to it.
- `src/model_client/openai.rs` — `OpenAIEmbeddingClient`. Its
  `generate_embeddings` (lines 202–237) already takes `&[String]`, serializes
  the whole slice as `"input"`, and **already sorts the response by the
  per-embedding `index` field** (lines 229–231) — no response-struct change is
  needed. Its `embedding_endpoint` (lines 190–192) is **hardcoded** — this
  plan adds the `OPENAI_BASE_URL` override (plan 003's maintenance notes
  deliberately deferred wiremocking the embeddings client because that
  override was out of its scope) so the wiremock tests can point it at a
  mock server.
- `tests/embedding_batch_tests.rs` — does not exist; created by this plan.
- `tests/provider_contract_tests.rs` — created by plan 003; used only as the
  structural exemplar for the env-lock pattern. Do NOT modify it.
- `Cargo.toml` — plan 003 adds `wiremock = "0.6"` under `[dev-dependencies]`.
  This plan adds no dependencies.

### Key excerpts (verified against the working tree at `afc78da`)

The env-override idiom to copy, `src/model_client/mod.rs:33-41`:

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

The per-row implementation being replaced, `src/model_client/mod.rs:488-512`:

```rust
async fn embed_one<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    text: &String,
) -> Option<Vec<f64>> {
    match client.generate_embeddings(http_client(), std::slice::from_ref(text)).await {
        Ok(embeddings) => embeddings.into_iter().next(),
        Err(e) => {
            eprintln!("Error generating embedding from {}: {}", client.provider_name(), e);
            None
        }
    }
}

/// Parallel embedding generation function with bounded concurrency
pub async fn fetch_embeddings_generic<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    texts: &[String]
) -> Vec<Option<Vec<f64>>> {
    let requests: Vec<_> = texts.iter().map(|text| embed_one(client, text)).collect();

    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}
```

(`embed_one` is KEPT — it becomes the fallback path. Only
`fetch_embeddings_generic` changes shape, plus two new private helpers.)

The batch-capable request path that is already correct,
`src/model_client/openai.rs:202-237` (do not change this function):

```rust
    async fn generate_embeddings(
        &self,
        client: &Client,
        texts: &[String],
    ) -> Result<Vec<Vec<f64>>, ModelClientError> {
        ...
        let request_body = OpenAIEmbeddingRequest {
            input: texts,
            model: &self.model,
            encoding_format: "float",
        };
        ...
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
    }
```

The hardcoded embedding endpoint you will make overridable,
`src/model_client/openai.rs:190-192`:

```rust
    fn embedding_endpoint(&self) -> String {
        "https://api.openai.com/v1/embeddings".to_string()
    }
```

The base-URL override pattern to copy (same file),
`src/model_client/openai.rs:57-61`:

```rust
    fn api_endpoint(&self) -> String {
        let base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        format!("{}/v1/chat/completions", base.trim_end_matches('/'))
    }
```

### Conventions and constraints

- Only OpenAI implements embeddings. Plan 005 makes `create_embedding_client`
  reject every other provider with `ModelClientError::Unsupported`, so this
  plan's batching only needs to be correct for OpenAI. Do not add embedding
  support for other providers.
- Env-var mutation in tests is process-global; the repo pattern is a
  poisoning-tolerant static mutex held for the whole test body — see
  `tests/model_client_tests.rs:16-20` and its copy in
  `tests/provider_contract_tests.rs` (from plan 003). Match it. Do NOT call
  `dotenvy::dotenv()` in the new test file (it could load real keys/base-URLs
  from a developer's `.env`).
- Edition is 2021 (Cargo.toml line 4), so `std::env::set_var` is a safe
  function here.
- `OPENAI_BASE_URL` semantics (base host only; client appends the path;
  `trim_end_matches('/')`) are shared with the chat endpoint and the local
  MLX server backend (README.md:231) — the embeddings override must follow
  the exact same semantics.
- Documented tradeoffs you must not disturb: pyo3 pinned <0.29
  (`.cargo/audit.toml`), reqwest rustls-native-roots / ring choices
  (Cargo.toml comments), mlx-lm transformers pin (pyproject.toml comments).
  Nothing in this plan touches them.
- Commit style: sentence-case imperative summaries, e.g.
  `Refresh cargo-audit ignore rationale for the pyo3 CVEs` (from `git log`).

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`.
No live API key is required by any verification in this plan.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile lib + tests | `cargo check --tests` | exit 0 |
| Format | `cargo fmt --all` then `cargo fmt --all -- --check` | exit 0, no output |
| Lint (repo standard) | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0, no warnings |
| Rust unit tests | `cargo test --lib` | exit 0, `0 failed` |
| New tests, guaranteed keyless | `env -u OPENAI_API_KEY -u OPENAI_BASE_URL -u POLAR_LLAMA_EMBED_BATCH_SIZE -u POLAR_LLAMA_MAX_CONCURRENCY cargo test --test embedding_batch_tests` | `test result: ok. 3 passed` |
| Contract-test regression (after 003) | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` | `test result: ok.`, 0 failed |
| Venv (only if `.venv` missing) | `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt maturin` | exit 0 |
| Build + install plugin | `source .venv/bin/activate && maturin develop` | exit 0, ends with `Installed polar-llama-...` |
| CI-safe Python regression | `source .venv/bin/activate && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | exit 0 (skips are fine) |

Do not run bare `cargo test` if a `.env` with real keys exists —
`tests/model_client_tests.rs` calls `dotenvy::dotenv()` and would hit live
APIs. Use the `env -u ...` forms above.

## Scope

**In scope** (the only files you may modify/create):

- `src/model_client/mod.rs` — add `embed_batch_size()` next to
  `max_concurrency()`; add private helpers `embed_chunk` /
  `embed_chunk_singly`; rework the body of `fetch_embeddings_generic`; append
  one unit test to the `#[cfg(test)] mod tests` module (created by plan 005).
- `src/model_client/openai.rs` — the `embedding_endpoint` body ONLY (add the
  `OPENAI_BASE_URL` override).
- `tests/embedding_batch_tests.rs` — create.
- `plans/README.md` — status row only, if the file exists.

**Out of scope** (do NOT touch, even though they look related):

- `generate_embeddings` in `src/model_client/openai.rs` — it already batches
  and already sorts by the response `index` field; no change needed or allowed.
- `embed_one`'s signature/body — it is kept verbatim as the fallback unit.
- Non-OpenAI providers — none support embeddings (enforced by plan 005).
- `src/utils.rs`, `src/expressions.rs`, `polar_llama/__init__.py` — the
  signature of `fetch_embeddings_generic` does not change, so no caller
  changes; expression-level API changes are explicitly out of the finding's
  scope.
- `tests/provider_contract_tests.rs`, `tests/model_client_tests.rs`,
  `tests/test_embeddings.py` — regression-only; do not edit.
- Retry/backoff for transient errors (rate limits etc.) — that is
  plans/008-retry-rate-limit.md; the chunk fallback here is failure isolation,
  not a retry policy.
- `Cargo.toml` — no new dependencies (wiremock arrives via plan 003).
- README / docs — `POLAR_LLAMA_MAX_CONCURRENCY` is likewise undocumented in
  README today; documenting both is deferred (see Maintenance notes).

## Git workflow

- Branch: `advisor/014-batch-embeddings` (branched from `main` AFTER plans 003
  and 005 have landed on it — see Step 0).
- One commit per step or logical unit; sentence-case imperative summaries,
  e.g. `Batch embedding requests with per-chunk single-row fallback`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Verify the dependencies have landed

This plan requires plans 003 and 005 to be merged first.

**Verify** (all three must hold, else STOP):

1. `grep -n "wiremock" Cargo.toml` → one hit, under `[dev-dependencies]`
   (plan 003 landed).
2. `grep -n "Unsupported" src/model_client/mod.rs` → at least 2 hits (enum
   variant + Display arm; plan 005 landed).
3. `grep -A3 "pub fn create_embedding_client" src/model_client/mod.rs` →
   the return type is `Result<Box<dyn EmbeddingClient + Send + Sync>, ModelClientError>`
   (plan 005's Result-returning factory).

Then run the drift check from the header and confirm only the expected drift
described there.

### Step 1: Add `embed_batch_size()` to `src/model_client/mod.rs`

Insert immediately after `max_concurrency()` (the function quoted in "Current
state"; at `afc78da` it ends at line 41):

```rust
/// Number of texts sent per embeddings HTTP request.
/// OpenAI's /v1/embeddings accepts up to 2048 inputs per call; 512 is a
/// conservative default that keeps request bodies small while cutting a
/// 10k-row column from 10k round-trips to ~20.
/// Override with the POLAR_LLAMA_EMBED_BATCH_SIZE environment variable
/// (same idiom as POLAR_LLAMA_MAX_CONCURRENCY above).
fn embed_batch_size() -> usize {
    std::env::var("POLAR_LLAMA_EMBED_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(512)
}
```

The `.filter(|v| *v > 0)` is load-bearing: `texts.chunks(0)` panics, so a
zero/invalid value must fall back to 512.

**Verify**: `cargo check` → exit 0 (an "unused function" warning is fine until
Step 3 wires it in; `cargo check` does not fail on warnings).

### Step 2: Add the `OPENAI_BASE_URL` override to `embedding_endpoint`

In `src/model_client/openai.rs`, replace the `embedding_endpoint` body (lines
190–192 at `afc78da`, inside `impl EmbeddingClient for OpenAIEmbeddingClient`)
with the same pattern as `api_endpoint` (openai.rs:57-61, quoted in "Current
state"):

```rust
    fn embedding_endpoint(&self) -> String {
        // OPENAI_BASE_URL overrides the host for proxies/tests
        // (same semantics as api_endpoint above: base host only,
        // client appends the path).
        let base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        format!("{}/v1/embeddings", base.trim_end_matches('/'))
    }
```

No other line in `openai.rs` changes. `generate_embeddings`, the request/
response structs, and the index-sorting logic are untouched.

**Verify**: `cargo check` → exit 0, and
`grep -c "OPENAI_BASE_URL" src/model_client/openai.rs` → `2` (chat +
embeddings).

### Step 3: Rework `fetch_embeddings_generic` into chunked batching with retry-shrink fallback

In `src/model_client/mod.rs`, keep `embed_one` exactly as it is (excerpt in
"Current state"). Immediately after it, add the two helpers, and replace the
body of `fetch_embeddings_generic`:

```rust
/// Embed every text in `chunk` with individual single-item requests.
///
/// This is the fallback path when a batched request fails: it preserves
/// per-row isolation (only the truly-bad rows become None) at the cost of
/// one request per row — for this chunk only. Concurrency within the
/// fallback is capped at max_concurrency(); since failing chunks are the
/// rare path, the theoretical worst case (several failing chunks each
/// running a capped fallback) is accepted for simplicity.
async fn embed_chunk_singly<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    chunk: &[String],
) -> Vec<Option<Vec<f64>>> {
    let requests: Vec<_> = chunk.iter().map(|text| embed_one(client, text)).collect();
    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}

/// Embed one chunk of texts with a single batched HTTP request.
///
/// Failure semantics ("retry-shrink"): a failed batched request would
/// otherwise null the entire chunk, so on any chunk-level failure — an HTTP
/// error, or a success response whose embedding count does not match the
/// input count — the chunk is re-issued as per-item requests, preserving the
/// pre-batching behavior where one bad row fails alone. The happy path is
/// still exactly one request per chunk.
async fn embed_chunk<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    chunk: &[String],
) -> Vec<Option<Vec<f64>>> {
    match client.generate_embeddings(http_client(), chunk).await {
        Ok(embeddings) if embeddings.len() == chunk.len() => {
            embeddings.into_iter().map(Some).collect()
        }
        Ok(embeddings) => {
            eprintln!(
                "Embedding batch from {} returned {} vectors for {} inputs; retrying rows individually",
                client.provider_name(),
                embeddings.len(),
                chunk.len()
            );
            embed_chunk_singly(client, chunk).await
        }
        Err(e) => {
            eprintln!(
                "Error generating embedding batch from {}: {}; retrying rows individually",
                client.provider_name(),
                e
            );
            embed_chunk_singly(client, chunk).await
        }
    }
}

/// Parallel embedding generation with provider-batch requests.
///
/// Texts are chunked into embed_batch_size() groups; each chunk is one HTTP
/// request (OpenAI's /v1/embeddings accepts arrays of inputs), and chunks
/// are driven concurrently up to max_concurrency(). `buffered` preserves
/// input order and `chunks` is order-stable, so flattening the per-chunk
/// results keeps the output aligned with the input rows.
pub async fn fetch_embeddings_generic<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    texts: &[String],
) -> Vec<Option<Vec<f64>>> {
    let chunk_requests: Vec<_> = texts
        .chunks(embed_batch_size())
        .map(|chunk| embed_chunk(client, chunk))
        .collect();

    futures::stream::iter(chunk_requests)
        .buffered(max_concurrency())
        .collect::<Vec<Vec<Option<Vec<f64>>>>>()
        .await
        .into_iter()
        .flatten()
        .collect()
}
```

Notes for the executor:

- The public signature of `fetch_embeddings_generic` is unchanged, so its
  caller (`fetch_embeddings_with_provider` in `src/utils.rs`) compiles as-is.
- Empty input still yields an empty output (`chunks` on an empty slice yields
  nothing).
- A single-row chunk that fails is retried once more via the fallback (2
  requests total for that row); that mild double-cost on the rare path is
  intentional and simpler than special-casing chunk length 1.

**Verify**: `cargo check --tests` → exit 0. Then
`RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0.

### Step 4: Append the unit test for `embed_batch_size`

At the bottom of `src/model_client/mod.rs`, inside the existing
`#[cfg(test)] mod tests { ... }` module created by plan 005, append (a single
test function so the process-global env var has exactly one mutator in the
lib-test binary — no lock needed):

```rust
    #[test]
    fn embed_batch_size_default_and_override() {
        // This is the only lib test touching this env var; keeping all
        // mutations in one #[test] avoids cross-thread races.
        std::env::remove_var("POLAR_LLAMA_EMBED_BATCH_SIZE");
        assert_eq!(embed_batch_size(), 512);

        std::env::set_var("POLAR_LLAMA_EMBED_BATCH_SIZE", "100");
        assert_eq!(embed_batch_size(), 100);

        // Zero and garbage fall back to the default (chunks(0) would panic).
        std::env::set_var("POLAR_LLAMA_EMBED_BATCH_SIZE", "0");
        assert_eq!(embed_batch_size(), 512);
        std::env::set_var("POLAR_LLAMA_EMBED_BATCH_SIZE", "not-a-number");
        assert_eq!(embed_batch_size(), 512);

        std::env::remove_var("POLAR_LLAMA_EMBED_BATCH_SIZE");
    }
```

**Verify**:
`env -u POLAR_LLAMA_EMBED_BATCH_SIZE cargo test --lib embed_batch_size` →
`test result: ok. 1 passed`. Then `cargo test --lib` → exit 0, `0 failed`
(plan 005's tests still pass).

### Step 5: Create `tests/embedding_batch_tests.rs`

Create the file with this exact scaffolding (env-lock pattern copied from
`tests/provider_contract_tests.rs`, which plan 003 modeled on
`tests/model_client_tests.rs:16-20`):

```rust
//! Keyless wiremock tests for embedding request batching
//! (fetch_embeddings_generic): chunk-count, order preservation via the
//! response `index` field, and the per-chunk single-row fallback.
//! No API keys and no live network access are required.

use polar_llama::model_client::openai::OpenAIEmbeddingClient;
use polar_llama::model_client::fetch_embeddings_generic;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Serializes tests that mutate process-global env vars
/// (OPENAI_BASE_URL, POLAR_LLAMA_EMBED_BATCH_SIZE). Same pattern as
/// ENV_LOCK in tests/provider_contract_tests.rs — poisoning is ignored so
/// one failing test doesn't cascade.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Responds to POST /v1/embeddings by deriving each embedding from its
/// input text: "<prefix>-<N>" -> the 1-element vector [N]. The data array
/// is returned in REVERSED order with correct `index` values, so these
/// tests also pin the index-based reordering in
/// OpenAIEmbeddingClient::generate_embeddings (openai.rs "Sort by index").
/// Any request whose inputs contain the literal "BAD" gets a 500 — so a
/// batch containing the poison row fails as a whole, and the single-item
/// fallback request for "BAD" also fails, while its neighbors succeed.
struct EchoEmbeddings;

impl Respond for EchoEmbeddings {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value =
            serde_json::from_slice(&request.body).expect("request body is JSON");
        let inputs: Vec<String> = body["input"]
            .as_array()
            .expect("input is an array")
            .iter()
            .map(|v| v.as_str().expect("input items are strings").to_string())
            .collect();

        if inputs.iter().any(|t| t == "BAD") {
            return ResponseTemplate::new(500).set_body_string("boom");
        }

        let mut data: Vec<Value> = inputs
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let n: f64 = text
                    .rsplit('-')
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("inputs look like prefix-N");
                json!({ "object": "embedding", "index": i, "embedding": [n] })
            })
            .collect();
        data.reverse(); // client must sort by `index`, not trust array order

        ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
        }))
    }
}
```

Then add the three tests. Every test takes `let _env = env_guard();` as its
first statement and removes both env vars before returning.

Test 1 — request count is exactly `ceil(N / batch_size)`:

```rust
#[tokio::test]
async fn thousand_texts_batch_100_sends_exactly_10_requests() {
    let _env = env_guard();
    let server = MockServer::start().await;
    std::env::set_var("OPENAI_BASE_URL", server.uri());
    std::env::set_var("POLAR_LLAMA_EMBED_BATCH_SIZE", "100");

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(EchoEmbeddings)
        .expect(10) // 1000 texts / batch 100 — verified on MockServer drop
        .mount(&server)
        .await;

    let texts: Vec<String> = (0..1000).map(|i| format!("text-{i}")).collect();
    let client = OpenAIEmbeddingClient::new_with_model("text-embedding-3-small");
    let results = fetch_embeddings_generic(&client, &texts).await;

    assert_eq!(results.len(), 1000);
    for (i, row) in results.iter().enumerate() {
        assert_eq!(
            row.as_deref(),
            Some(&[i as f64][..]),
            "row {i} missing or out of order"
        );
    }

    std::env::remove_var("POLAR_LLAMA_EMBED_BATCH_SIZE");
    std::env::remove_var("OPENAI_BASE_URL");
}
```

Test 2 — a failing chunk falls back to singles; only the poison row is None:

```rust
#[tokio::test]
async fn failing_chunk_falls_back_to_singles_and_nulls_only_bad_row() {
    let _env = env_guard();
    let server = MockServer::start().await;
    std::env::set_var("OPENAI_BASE_URL", server.uri());
    std::env::set_var("POLAR_LLAMA_EMBED_BATCH_SIZE", "100");

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(EchoEmbeddings)
        .expect(4) // 1 failed batch of 3 + 3 single-item fallback requests
        .mount(&server)
        .await;

    let texts = vec!["ok-0".to_string(), "BAD".to_string(), "ok-2".to_string()];
    let client = OpenAIEmbeddingClient::new_with_model("text-embedding-3-small");
    let results = fetch_embeddings_generic(&client, &texts).await;

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].as_deref(), Some(&[0.0][..]));
    assert_eq!(results[1], None, "only the truly-bad row is null");
    assert_eq!(results[2].as_deref(), Some(&[2.0][..]));

    std::env::remove_var("POLAR_LLAMA_EMBED_BATCH_SIZE");
    std::env::remove_var("OPENAI_BASE_URL");
}
```

(The failing rows print `Error ... boom` lines to stderr from the `eprintln!`s
in `embed_chunk`/`embed_one` — expected noise, not a failure.)

Test 3 — order preserved across multiple chunks including a partial final
chunk:

```rust
#[tokio::test]
async fn order_preserved_across_chunks_with_partial_final_chunk() {
    let _env = env_guard();
    let server = MockServer::start().await;
    std::env::set_var("OPENAI_BASE_URL", server.uri());
    std::env::set_var("POLAR_LLAMA_EMBED_BATCH_SIZE", "2");

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(EchoEmbeddings)
        .expect(3) // 5 texts / batch 2 -> chunks of 2, 2, 1
        .mount(&server)
        .await;

    let texts: Vec<String> = (0..5).map(|i| format!("t-{i}")).collect();
    let client = OpenAIEmbeddingClient::new_with_model("text-embedding-3-small");
    let results = fetch_embeddings_generic(&client, &texts).await;

    let flat: Vec<f64> = results
        .iter()
        .map(|r| r.as_ref().expect("all rows succeed")[0])
        .collect();
    assert_eq!(flat, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    std::env::remove_var("POLAR_LLAMA_EMBED_BATCH_SIZE");
    std::env::remove_var("OPENAI_BASE_URL");
}
```

**Verify**:
`env -u OPENAI_API_KEY -u OPENAI_BASE_URL -u POLAR_LLAMA_EMBED_BATCH_SIZE -u POLAR_LLAMA_MAX_CONCURRENCY cargo test --test embedding_batch_tests`
→ `test result: ok. 3 passed; 0 failed`.

### Step 6: Full verification sweep

Run, in order:

1. `cargo fmt --all` then `cargo fmt --all -- --check` → exit 0
2. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0
3. `cargo test --lib` → exit 0, `0 failed`
4. `env -u OPENAI_API_KEY -u OPENAI_BASE_URL -u POLAR_LLAMA_EMBED_BATCH_SIZE -u POLAR_LLAMA_MAX_CONCURRENCY cargo test --test embedding_batch_tests` → 3 passed
5. `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` → exit 0 (plan 003's suite unaffected)
6. `source .venv/bin/activate && maturin develop` → exit 0
7. `source .venv/bin/activate && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` → exit 0 (key-gated tests skip; that is expected)

**Verify**: all seven commands succeed as stated.

## Test plan

- New Rust unit test in `src/model_client/mod.rs` `#[cfg(test)] mod tests`
  (Step 4): `embed_batch_size_default_and_override` — default 512, valid
  override honored, zero/garbage fall back (guards the `chunks(0)` panic).
- New integration tests in `tests/embedding_batch_tests.rs` (Step 5;
  structural exemplar: `tests/provider_contract_tests.rs` from plan 003):
  1. `thousand_texts_batch_100_sends_exactly_10_requests` — request count ==
     ceil(N/batch) enforced by wiremock `.expect(10)`, plus all 1000 values
     correct and in input order.
  2. `failing_chunk_falls_back_to_singles_and_nulls_only_bad_row` — the
     retry-shrink regression test: exactly 4 requests, only the poison row is
     `None`.
  3. `order_preserved_across_chunks_with_partial_final_chunk` — multi-chunk
     order with a partial trailing chunk; combined with the responder's
     reversed data array, this pins the `index`-based reordering.
- Existing suites are regression gates: `cargo test --lib` (plan 005's
  embedding-factory tests), `provider_contract_tests` (plan 003), and the
  CI-safe pytest run (the Python `embedding_async` path compiles against the
  unchanged `fetch_embeddings_generic` signature).
- Verification: commands in Step 6.

## Done criteria

Machine-checkable. ALL must hold (run from the repo root):

- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `cargo test --lib` exits 0 and includes
      `embed_batch_size_default_and_override` passing
- [ ] `env -u OPENAI_API_KEY -u OPENAI_BASE_URL -u POLAR_LLAMA_EMBED_BATCH_SIZE -u POLAR_LLAMA_MAX_CONCURRENCY cargo test --test embedding_batch_tests`
      exits 0 with `3 passed; 0 failed`
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests`
      exits 0
- [ ] `grep -c "OPENAI_BASE_URL" src/model_client/openai.rs` prints `2`
- [ ] `grep -n "POLAR_LLAMA_EMBED_BATCH_SIZE" src/model_client/mod.rs` shows
      the `embed_batch_size` reader (plus the doc comment and unit test)
- [ ] `grep -n 'texts.iter().map(|text| embed_one' src/model_client/mod.rs`
      returns no matches (the per-row driver is gone from
      `fetch_embeddings_generic`; `embed_one` itself remains as the fallback
      unit inside `embed_chunk_singly`)
- [ ] `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
      exits 0 (after `maturin develop`)
- [ ] `git status --porcelain` shows changes only to `src/model_client/mod.rs`,
      `src/model_client/openai.rs`, `tests/embedding_batch_tests.rs` (and
      `plans/README.md` if it exists)
- [ ] `plans/README.md` status row updated, if that file exists

## STOP conditions

Stop and report back (do not improvise) if:

- Step 0 fails: `wiremock` is not in `[dev-dependencies]` (plan 003 not
  landed), or `create_embedding_client` does not return a `Result` /
  `ModelClientError` has no `Unsupported` variant (plan 005's
  Result-returning factory not landed). This plan must be rebased on both.
- The drift check shows changes beyond the expected plan-003/005 drift
  described in the header — in particular, if `embed_one` or
  `fetch_embeddings_generic` no longer match the "Current state" excerpts, or
  `openai.rs` has changed at all.
- `OpenAIEmbeddingData` in `src/model_client/openai.rs` no longer has an
  `index: usize` field, or `generate_embeddings` no longer sorts by it —
  the order-preservation guarantee this plan relies on has been removed;
  re-adding it is a behavior question for the advisor, not an improvisation.
- The wiremock tests pass only with `--test-threads=1` — the ENV_LOCK pattern
  is not serializing env access; report rather than hard-coding
  `--test-threads=1`.
- Making any test pass appears to require changing `generate_embeddings`,
  `src/utils.rs`, `src/expressions.rs`, or any other out-of-scope file.
- Any step's verification fails twice after a reasonable fix attempt.
- `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` fails on pre-existing
  code unrelated to your changes.

## Maintenance notes

- **Interaction with plans/008-retry-rate-limit.md**: the per-chunk fallback
  here is failure *isolation*, not a retry policy. When 008 adds
  backoff/retry to the send path, a rate-limited (429) chunk will be retried
  by 008's machinery before the fallback fires; make sure 008's retries wrap
  `generate_embeddings` so a chunk isn't immediately shattered into
  N single requests on a transient 429 (which would make rate limiting
  worse). Revisit `embed_chunk`'s `Err` arm at that point.
- **When non-OpenAI embedding providers land** (deferred direction item noted
  in plan 005), each provider has a different max batch size (OpenAI 2048;
  others vary). The right shape then is a `max_batch_size()` method on the
  `EmbeddingClient` trait with the env var as a clamp — today a single global
  default is fine because only OpenAI exists.
- **Reviewer should scrutinize**: (a) order preservation — it relies on
  `chunks` + `buffered` (order-preserving) + `flatten`, and on the `index`
  sort inside `generate_embeddings`; test 3's reversed responder is the pin
  for the latter. (b) The fallback's concurrency amplification comment in
  `embed_chunk_singly` — accepted worst case, documented in code. (c) The
  `embedding_endpoint` override is byte-for-byte in the style of
  `api_endpoint` with no auth changes.
- **Deliberately deferred**: README documentation for
  `POLAR_LLAMA_EMBED_BATCH_SIZE` (its sibling `POLAR_LLAMA_MAX_CONCURRENCY`
  is also undocumented in README today — document both together in a docs
  pass); raising the default above 512 (needs real-workload latency/body-size
  data); a `dimensions` request parameter for `text-embedding-3-*` models
  (unrelated feature).
