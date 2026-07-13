# Plan 004: Cap the cache-warming fan-out at max_concurrency instead of unbounded join_all

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index. (If `plans/README.md` does not exist, skip that update —
> the advisor maintains the index.)
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/utils.rs src/model_client/mod.rs Cargo.toml`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. Exception: if
> plans/003-provider-http-mock-tests.md already landed, `Cargo.toml` will show
> a `[dev-dependencies]` addition (wiremock) — that specific change is
> expected, not drift (see Step 3).

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/003-provider-http-mock-tests.md (soft — it introduces the
  wiremock mock-server dev-dependency and test precedent; Step 3 of this plan
  has a fallback that adds wiremock itself, so this plan is executable today
  even if 003 has not landed)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

When prompt caching is enabled (`cache=true`), rows sharing a cacheable prefix
are grouped, the first row's request is sent serially to warm the provider's
prompt cache, and then **every remaining row in the group is fired
simultaneously** via `futures::future::join_all` with no concurrency bound.
A 10,000-row DataFrame with a shared system prompt — exactly the workload
caching is designed for — fires ~10,000 concurrent HTTP requests: provider
429 storms, local file-descriptor exhaustion, and timeouts. The normal
(non-cache) batch path already bounds concurrency at `max_concurrency()`
(env `POLAR_LLAMA_MAX_CONCURRENCY`, default 64). This plan applies the same
bound to the cache-warming fan-out, preserving result order.

## Current state

Relevant files:

- `src/utils.rs` — blocking/async fetch helpers; contains
  `fetch_with_cache_warming` (lines 148–256) with the unbounded `join_all` at
  line 251. **This is where the fix goes.**
- `src/model_client/mod.rs` — provider abstraction; `max_concurrency()`
  (lines 33–41, currently private) and the bounded batch pattern to copy
  (`run_batch`, lines 397–417; same pattern again at line 509 for embeddings).
- `src/expressions.rs` — the only caller of `fetch_with_cache_warming`
  (lines 433–438), inside the cache-group processing loop. Read-only context;
  do not modify.
- `Cargo.toml` — `[dev-dependencies]` (lines 46–48) currently has only
  `dotenvy` and `tokio-test`; wiremock must be present for the new test
  (added by plan 003, or by Step 3's fallback).

### Excerpt 1 — the bug, `src/utils.rs:199-253` (abridged; middle of the closure omitted)

```rust
    let mut results = vec![first_result];

    // Step 2: Process remaining requests in parallel (should hit cache)
    if message_arrays.len() > 1 {
        let remaining_futures: Vec<_> = message_arrays[1..]
            .iter()
            .map(|msgs| {
                let client = create_client(provider, model);
                let reqwest_client = reqwest_client.clone();
                let messages = msgs.clone();
                let schema_owned = response_schema.map(|s| s.to_string());
                let model_name_owned = response_model_name.map(|s| s.to_string());

                async move {
                    // ... sends one request, validates schema, maps errors
                    // to Some(create_error_response(...)) — leave unchanged
                }
            })
            .collect();

        let remaining_results = futures::future::join_all(remaining_futures).await;   // line 251 — UNBOUNDED
        results.extend(remaining_results);
    }

    results
}
```

`src/utils.rs` imports today (lines 1–4) — note there is **no**
`futures::StreamExt` import yet:

```rust
use polars::prelude::*;
use std::sync::LazyLock;
use tokio::runtime::Runtime;
use crate::model_client::{self, Provider, create_client, create_embedding_client, Message, ModelClientError};
```

### Excerpt 2 — the pattern to copy, `src/model_client/mod.rs:397-417`

```rust
async fn run_batch<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>,
    errors_as_json: bool,
) -> Vec<Option<String>> {
    let schema_check = SchemaCheck::compile(schema);

    // Collect the futures eagerly so the stream holds concrete future values;
    // requests still only run when polled, bounded by `buffered`.
    let requests: Vec<_> = message_arrays
        .iter()
        .map(|messages| run_one(client, messages, schema, model_name, &schema_check, errors_as_json))
        .collect();

    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}
```

### Excerpt 3 — the function to expose, `src/model_client/mod.rs:33-41`

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

### Excerpt 4 — the call site (context only, do not modify), `src/expressions.rs:433-438`

```rust
            run_async(async move {
                crate::utils::fetch_with_cache_warming(
                    &msgs, provider, &m,
                    schema.as_deref(), model_name.as_deref()
                ).await
            })
```

### Facts that make the test possible without live API keys

- `OpenAIClient::api_endpoint()` honors the `OPENAI_BASE_URL` env var
  (`src/model_client/openai.rs:57-61`):
  `format!("{}/v1/chat/completions", base.trim_end_matches('/'))`.
- The API key comes from `OPENAI_API_KEY` via `unwrap_or_default()`
  (`src/model_client/mod.rs:231-233`) — any dummy value works against a mock.
- `OpenAIClient::parse_response` (`src/model_client/openai.rs:114-122`)
  deserializes `{ "id", "model", "choices": [{ "index", "message": { "role",
  "content" }, "finish_reason" }] }` and returns `choices[0].message.content`.
  The mock must return exactly this shape.
- Crate is `edition = "2021"` (`Cargo.toml:4`), so `std::env::set_var` is a
  safe function (no `unsafe` block needed in the test).
- `tokio` has features `rt-multi-thread`, `macros`, `sync`, `time`
  (`Cargo.toml:25`), so `#[tokio::test(flavor = "multi_thread")]` and
  `tokio::time::sleep` are available.

### Conventions

- Rust unit tests live in a `#[cfg(test)] mod tests { use super::*; ... }`
  block at the bottom of the source file — see `src/cache.rs:313` onward for
  the exemplar. Do NOT put the new test in `tests/` — `tests/model_client_tests.rs`
  there requires live API keys and is run only by a manual CI workflow.
- CI compiles with `RUSTFLAGS=-Dwarnings`, so any new warning (unused import,
  etc.) is a build failure.
- Documented constraint: `pyo3` is pinned `<0.29` by `pyo3-polars` (rationale
  in `.cargo/audit.toml`) — do not bump any pinned dependency versions while
  executing this plan.

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`
(adjust if checked out elsewhere).

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile check | `cargo check --all-features` | exit 0 |
| Format check | `cargo fmt --all -- --check` | exit 0, no output |
| Lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0, no warnings |
| Rust unit tests | `cargo test --lib` | all pass (15 pass at plan time; 16+ after this plan) |
| Run just the new test | `cargo test --lib cache_warming_fanout` | `1 passed` |

No Python build (`maturin develop`) is required — this plan touches only Rust
code paths whose behavior is verified by Rust unit tests. Never set real API
keys for any verification here.

## Scope

**In scope** (the only files you may modify):

- `src/utils.rs` — replace `join_all` with a bounded `buffered` stream; add
  the `futures::StreamExt` import; add the new `#[cfg(test)]` test module.
- `src/model_client/mod.rs` — visibility change ONLY: `fn max_concurrency()`
  → `pub(crate) fn max_concurrency()`. Nothing else in this file.
- `Cargo.toml` — ONLY if wiremock is not already a dev-dependency (i.e. plan
  003 has not landed): add `wiremock = "0.6"` under `[dev-dependencies]`.
  (`Cargo.lock` will update as a side effect of the dependency addition;
  that is expected.)

**Out of scope** (do NOT touch, even though they look related):

- `src/cache.rs` — cache grouping/breakpoint logic is correct and separate.
- `src/expressions.rs` — the call site is fine; the fix is inside
  `fetch_with_cache_warming`.
- The serial first-request design in `fetch_with_cache_warming` (warm the
  cache, then fan out) — keep it exactly as is; only the fan-out gets bounded.
- `src/model_client/{openai,anthropic,gemini,groq,bedrock}.rs` — provider
  clients unchanged.
- The default concurrency value (64) or the env var name — behavior of the
  normal batch path must not change.
- `tests/model_client_tests.rs` — live-API integration tests, manual workflow.
- Any dependency version bumps other than adding wiremock as a dev-dependency.

## Git workflow

- Branch: `advisor/004-bound-cache-warming-concurrency` (branch off `main`).
- Commit style: sentence-case imperative summary, matching repo history, e.g.
  `Bound the cache-warming fan-out at max_concurrency` (compare existing:
  `Refresh cargo-audit ignore rationale for the pyo3 CVEs`).
- One commit for the fix + test is fine; two commits (fix, then test) also fine.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Expose `max_concurrency` as `pub(crate)`

In `src/model_client/mod.rs` line 35, change:

```rust
fn max_concurrency() -> usize {
```

to:

```rust
pub(crate) fn max_concurrency() -> usize {
```

Do not change the function body, its doc comment, or anything else in the file.

**Verify**: `cargo check --all-features` → exit 0.

### Step 2: Replace `join_all` with a bounded `buffered` stream in `src/utils.rs`

2a. Add the import at the top of `src/utils.rs`, after the existing `use`
lines (after line 4):

```rust
use futures::StreamExt;
```

2b. In `fetch_with_cache_warming`, replace the two lines at 251–252:

```rust
        let remaining_results = futures::future::join_all(remaining_futures).await;
        results.extend(remaining_results);
```

with:

```rust
        // Bounded fan-out: same concurrency cap as the normal batch path
        // (model_client::run_batch). `buffered` (not `buffer_unordered`)
        // preserves input order — results feed positional row alignment
        // downstream in expressions.rs.
        let remaining_results: Vec<Option<String>> = futures::stream::iter(remaining_futures)
            .buffered(crate::model_client::max_concurrency())
            .collect()
            .await;
        results.extend(remaining_results);
```

Leave `remaining_futures` construction (the `.map(|msgs| { ... async move { ... } })
.collect()` block, lines 203–249) completely unchanged.

2c. Optionally update the comment at line 201 from
`// Step 2: Process remaining requests in parallel (should hit cache)` to
`// Step 2: Process remaining requests in parallel (should hit cache), bounded by max_concurrency()`.

**Verify**:
- `grep -n "join_all" src/utils.rs` → no output (exit code 1).
- `cargo check --all-features` → exit 0.
- `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0 (in
  particular, no unused-import warning for `futures::StreamExt` — it is used
  by `.buffered(...)`/`.collect()`).

### Step 3: Ensure wiremock is a dev-dependency

Check first: `grep -n "wiremock" Cargo.toml`.

- If it already appears under `[dev-dependencies]` (plan 003 landed), do
  nothing in this step.
- Otherwise, add one line to the `[dev-dependencies]` section of `Cargo.toml`
  (currently lines 46–48):

```toml
[dev-dependencies]
dotenvy = "0.15"
tokio-test = "0.4"
wiremock = "0.6"
```

**Verify**: `cargo check --all-features --tests` → exit 0 (wiremock downloads
and compiles).

### Step 4: Add the bounded-concurrency regression test to `src/utils.rs`

Append a `#[cfg(test)]` module at the bottom of `src/utils.rs` (after the
closing brace of `fetch_with_cache_warming`). Use this code as the target
shape — it is designed to be deterministic (the in-flight decrement fires at
50ms, strictly before the 100ms response delay, so the high-water mark can
only undercount, never overcount):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    /// Serializes tests that mutate process-global environment variables.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Mock responder that tracks the high-water mark of concurrently
    /// in-flight requests and echoes the last message content back in an
    /// OpenAI-shaped response body.
    struct ConcurrencyProbe {
        in_flight: Arc<AtomicUsize>,
        high_water: Arc<AtomicUsize>,
    }

    impl Respond for ConcurrencyProbe {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.high_water.fetch_max(now, Ordering::SeqCst);

            // Decrement strictly BEFORE the delayed response is delivered
            // (50ms < 100ms): the high-water mark can only undercount, so
            // the <= max_concurrency assertion below cannot flake high.
            let in_flight = Arc::clone(&self.in_flight);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
            });

            // Echo the last message's content so the test can assert that
            // response order matches input order.
            let echo = serde_json::from_slice::<serde_json::Value>(&request.body)
                .ok()
                .and_then(|v| {
                    v["messages"]
                        .as_array()
                        .and_then(|msgs| msgs.last())
                        .and_then(|m| m["content"].as_str().map(String::from))
                })
                .unwrap_or_default();

            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(100))
                .set_body_json(serde_json::json!({
                    "id": "mock",
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": echo },
                        "finish_reason": "stop"
                    }]
                }))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cache_warming_fanout_is_bounded_and_order_preserving() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let in_flight = Arc::new(AtomicUsize::new(0));
        let high_water = Arc::new(AtomicUsize::new(0));

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ConcurrencyProbe {
                in_flight: Arc::clone(&in_flight),
                high_water: Arc::clone(&high_water),
            })
            .mount(&server)
            .await;

        std::env::set_var("OPENAI_BASE_URL", server.uri());
        std::env::set_var("OPENAI_API_KEY", "test-key-not-real");
        std::env::set_var("POLAR_LLAMA_MAX_CONCURRENCY", "4");

        let message_arrays: Vec<Vec<Message>> = (0..32)
            .map(|i| {
                vec![Message {
                    role: "user".to_string(),
                    content: format!("row-{i}"),
                    cache_control: None,
                }]
            })
            .collect();

        let results = fetch_with_cache_warming(
            &message_arrays,
            Provider::OpenAI,
            "gpt-4o-mini",
            None,
            None,
        )
        .await;

        std::env::remove_var("POLAR_LLAMA_MAX_CONCURRENCY");
        std::env::remove_var("OPENAI_BASE_URL");
        std::env::remove_var("OPENAI_API_KEY");

        assert_eq!(results.len(), 32);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                result.as_deref(),
                Some(format!("row-{i}").as_str()),
                "row {i} missing or out of order"
            );
        }

        let peak = high_water.load(Ordering::SeqCst);
        assert!(
            peak <= 4,
            "in-flight high-water mark {peak} exceeded POLAR_LLAMA_MAX_CONCURRENCY=4"
        );
        assert!(
            peak >= 2,
            "requests never overlapped (peak {peak}) — fan-out appears serial"
        );
    }
}
```

Notes for the executor:

- `use super::*;` brings `Message`, `Provider`, and `fetch_with_cache_warming`
  into scope (they are declared/imported at the top of `src/utils.rs`).
- If plan 003 already landed a shared mock harness or an env-lock helper
  inside `src/` (check `grep -rn "MockServer\|ENV_LOCK" src/`), reuse its
  helpers instead of duplicating — but only if that requires no edits to
  out-of-scope files. Otherwise the self-contained module above is correct.
- The first of the 32 requests is the serial cache-warming request; the
  remaining 31 exercise the bounded fan-out. With the old `join_all` code the
  probe records a peak near 31, so this test fails before the fix and passes
  after it (you can sanity-check that by stashing the Step 2 change, but it
  is not required).

**Verify**: `cargo test --lib cache_warming_fanout` →
`test utils::tests::cache_warming_fanout_is_bounded_and_order_preserving ... ok`,
`1 passed`.

### Step 5: Full verification sweep

Run, in order, from the repo root:

1. `grep -n "join_all" src/utils.rs` → no output.
2. `grep -rn "join_all" src/` → no matches anywhere in `src/` (at plan time
   `src/utils.rs:251` was the only occurrence; if a NEW occurrence exists in
   a file you did not touch, that is drift — STOP).
3. `cargo fmt --all` then `cargo fmt --all -- --check` → exit 0.
4. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0.
5. `cargo test --lib` → all tests pass (15 pre-existing + the 1 new test;
   plan 003 may have added more).
6. `git status --porcelain` → only `src/utils.rs`, `src/model_client/mod.rs`,
   and (if Step 3 applied) `Cargo.toml` + `Cargo.lock` are modified. Note:
   at plan time the working tree already contained untracked files
   (`.DS_Store`, `test_cache_*.py`, `test_optional_llm.py` at repo root) —
   leave them alone; they are not yours.

## Test plan

- New test: `utils::tests::cache_warming_fanout_is_bounded_and_order_preserving`
  in `src/utils.rs` (Step 4). Covers:
  - the regression this plan fixes: with `POLAR_LLAMA_MAX_CONCURRENCY=4` and
    32 rows, the observed in-flight high-water mark is ≤ 4 (fails at ~31 on
    the old `join_all` code);
  - order preservation: `results[i]` echoes row `i`'s content (guards against
    someone later swapping `buffered` for `buffer_unordered`);
  - parallelism still happens: peak ≥ 2 (guards against accidentally
    serializing the fan-out).
- Structural pattern: `#[cfg(test)] mod tests` at file bottom, as in
  `src/cache.rs:313` onward.
- No existing test covers `fetch_with_cache_warming` without live keys, so no
  existing test needs updating.
- Verification: `cargo test --lib` → all pass, including the 1 new test.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "join_all" src/utils.rs` returns nothing (exit code 1)
- [ ] `grep -n "pub(crate) fn max_concurrency" src/model_client/mod.rs` returns one match
- [ ] `grep -n "buffered(crate::model_client::max_concurrency())" src/utils.rs` returns one match
- [ ] `cargo test --lib` exits 0; `cargo test --lib cache_warming_fanout` reports `1 passed`
- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `git status --porcelain` shows changes only to `src/utils.rs`,
      `src/model_client/mod.rs`, and (only if Step 3 added wiremock)
      `Cargo.toml`/`Cargo.lock`
- [ ] No environment variable containing a real API key was set at any point
- [ ] `plans/README.md` status row updated (skip if the file does not exist or
      a reviewer told you they maintain the index)

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows changes to `src/utils.rs` or `src/model_client/mod.rs`
  since `afc78da` AND the live code no longer matches the "Current state"
  excerpts (in particular: `fetch_with_cache_warming`'s signature at
  `src/utils.rs:148-154`, the `join_all` call at line 251, or the private
  `fn max_concurrency()` at `src/model_client/mod.rs:35`).
- The call site in `src/expressions.rs:433-438` differs materially from
  Excerpt 4 (e.g. `fetch_with_cache_warming` gained parameters or new callers
  that depend on unbounded behavior).
- `wiremock = "0.6"` fails to resolve or compile against this crate's
  dependency tree (e.g. a hyper/tokio version conflict) — do not start bumping
  other dependencies to make it fit; report instead.
- The new test fails or flakes twice in a row after one reasonable timing
  adjustment (you may widen the gap once: decrement sleep 50ms → 25ms,
  response delay 100ms → 200ms). Do not weaken the `peak <= 4` assertion.
- The fix appears to require touching any out-of-scope file (especially
  `src/cache.rs` or `src/expressions.rs`).
- `cargo test --lib` was failing BEFORE your changes (baseline broken — at
  plan time it passed with 15 tests).

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **Reviewer focus**: (1) `buffered`, not `buffer_unordered` — order carries
  positional row alignment through `expressions.rs`'s
  `group_results`→`original_idx` mapping; (2) the `remaining_futures` closure
  bodies are byte-identical to before — only the driver changed; (3) the
  visibility change on `max_concurrency` is `pub(crate)`, not `pub` — it must
  not become part of the public rlib API.
- The cache-warming path now shares the `POLAR_LLAMA_MAX_CONCURRENCY` knob
  with the normal batch path (`run_batch`, mod.rs:397-417) and the embeddings
  path (`fetch_embeddings_generic`, mod.rs:502-512). If a per-call concurrency
  parameter is ever added to the Python API, thread it through all three call
  sites together.
- Cache groups are processed sequentially in `src/expressions.rs` (comment at
  lines 409–411), so the effective global bound with caching is
  `max_concurrency()` per group at a time — no cross-group multiplication.
- The env-var mutation in the test is process-global; the `ENV_LOCK` mutex
  serializes it against future env-mutating tests in this crate. If plan 003's
  harness later centralizes an env lock or mock-server helpers, migrate this
  test's `ConcurrencyProbe` there (deferred out of this plan to keep its scope
  to one file).
- Deferred follow-up (out of scope here): retry/backoff on 429 responses.
  Bounding concurrency reduces 429s but does not handle them; error responses
  still surface as structured `api_error` JSON rows.
