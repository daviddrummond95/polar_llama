# Plan 007: One error contract — identical failures produce identical cell values regardless of cache path

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index (if `plans/README.md` does not exist, skip this).
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/model_client/mod.rs src/utils.rs src/expressions.rs tests/model_client_tests.rs tests/provider_contract_tests.rs CHANGELOG.md`
> Expected: `tests/provider_contract_tests.rs` WILL appear as added —
> plans/003-provider-http-mock-tests.md creates it after commit `afc78da`, and
> this plan depends on that. A `CHANGELOG.md` diff limited to new
> `[Unreleased]` entries is also acceptable. If `src/model_client/mod.rs`,
> `src/utils.rs`, `src/expressions.rs`, or `tests/model_client_tests.rs`
> changed, compare the "Current state" excerpts below against the live code
> before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/003-provider-http-mock-tests.md
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

The same logical failure (e.g. an HTTP 500 from the provider) encodes differently in the output column depending on which internal code path handled the row. Non-schema batch paths turn API errors into `None` (a null cell); schema paths and the cache-warming path turn the *same* error into a JSON string `{"_error": "api_error", "_details": ...}`. Worst case: with `cache=True`, whether a failed row comes back as null or as error-JSON depends on **how many rows happened to share its system prompt** — a multi-row cache group routes through the always-JSON cache-warming function, while a single-row group routes through the errors-as-null function. Users cannot write reliable failure-handling code against a nondeterministic encoding. After this plan lands, every request failure on every path produces the same `create_error_response` JSON, and a null output cell means exactly one thing: the input row was null.

## Current state

All excerpts verified against the working tree at `afc78da`.

### Files and roles

- `src/model_client/mod.rs` — core batch runner. `create_error_response` (lines 335–349) defines the error-JSON shape; `run_one` (lines 367–395) holds the `errors_as_json` branch; `run_batch` (lines 397–417) threads the flag; four public wrappers hardcode it (lines 433–468).
- `src/utils.rs` — blocking/async bridge functions. `fetch_with_cache_warming` (lines 148–256) ALWAYS emits error-JSON on failure (lines 195 and 244) — it never consults any flag. The thin wrappers `fetch_data_with_provider` (line 41), `fetch_data_message_arrays_with_provider` (line 60), and the `*_and_schema` variants (lines 106, 117) just call through to the mod.rs wrappers. **No code change is needed in this file.**
- `src/expressions.rs` — Polars expression entry points. `process_with_cache_groups` (lines 397–468) is where the inconsistency becomes user-visible: a multi-row cache group calls `fetch_with_cache_warming` (line 434, always JSON errors) while a single-row group without a schema calls `fetch_data_message_arrays_with_provider` (line 453, errors as null). The non-cached string path does the same split at lines 215–227. **No code change is needed in this file** — once the flag is removed at the mod.rs level, all routes converge.
- `tests/model_client_tests.rs` — live-API integration tests. Contains `test_error_handling_invalid_api_key` (lines 284–305), which asserts the OLD null-on-error contract and — unlike the other tests in that file — does NOT self-skip without keys (it deliberately sets an invalid key and calls the live endpoint). It must be updated or the full `cargo test` gate breaks after the change.
- `tests/provider_contract_tests.rs` — keyless wiremock harness created by plans/003-provider-http-mock-tests.md. Contains `batch_alignment_mixed_success_failure`, which asserts `results[1] == None` for a failed row — that assertion must be updated by this plan, and the new three-path equality test goes here.
- `CHANGELOG.md` — Keep-a-Changelog format; `## [Unreleased]` heading exists at line 8 (currently empty at planning time).

### The flag and every place it appears

`grep -rn "errors_as_json" src/` returns exactly these five hits, plus the four boolean literals at the wrapper call sites quoted below:

```
src/model_client/mod.rs:365  (doc comment)
src/model_client/mod.rs:373  (run_one parameter)
src/model_client/mod.rs:388  (the branch)
src/model_client/mod.rs:402  (run_batch parameter)
src/model_client/mod.rs:410  (run_batch -> run_one pass-through)
```

The divergent branch, `src/model_client/mod.rs:362-395`:

```rust
/// Core batch runner: sends all requests concurrently (bounded by
/// `max_concurrency`) over the shared HTTP client, preserving input order.
///
/// When `errors_as_json` is true, request errors are surfaced as structured
/// error JSON objects; otherwise they map to `None`.
async fn run_one<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[Message],
    schema: Option<&str>,
    model_name: Option<&str>,
    schema_check: &SchemaCheck,
    errors_as_json: bool,
) -> Option<String> {
    match client.send_request_structured(http_client(), messages, schema, model_name).await {
        Ok(response) => {
            match schema_check.validate(&response) {
                Ok(()) => Some(response),
                Err(validation_error) => Some(create_error_response(
                    "validation_failed",
                    &validation_error,
                    Some(&response),
                )),
            }
        },
        Err(e) => {
            eprintln!("Error fetching from {}: {}", client.provider_name(), e);
            if errors_as_json {
                Some(create_error_response("api_error", &e.to_string(), None))
            } else {
                None
            }
        }
    }
}
```

`run_batch` threads it, `src/model_client/mod.rs:397-417` (relevant lines):

```rust
async fn run_batch<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>,
    errors_as_json: bool,                                    // line 402
) -> Vec<Option<String>> {
    let schema_check = SchemaCheck::compile(schema);
    ...
        .map(|messages| run_one(client, messages, schema, model_name, &schema_check, errors_as_json))  // line 410
```

The four public wrappers hardcode the flag, `src/model_client/mod.rs:433-468`:

```rust
pub async fn fetch_data_generic<...>(...) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, None, None, false).await          // line 438 — errors -> None
}
pub async fn fetch_data_generic_with_schema<...>(...) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, schema, model_name, true).await   // line 449 — errors -> JSON
}
pub async fn fetch_data_generic_enhanced<...>(...) -> Vec<Option<String>> {
    run_batch(client, message_arrays, None, None, false).await           // line 457 — errors -> None
}
pub async fn fetch_data_generic_enhanced_with_schema<...>(...) -> Vec<Option<String>> {
    run_batch(client, message_arrays, schema, model_name, true).await    // line 467 — errors -> JSON
}
```

The cache-warming path always emits JSON regardless, `src/utils.rs:193-196` (and identically at lines 242–245 for the parallel remainder):

```rust
        Err(e) => {
            eprintln!("Error fetching from {} (cache warming): {}", provider.as_str(), e);
            Some(model_client::create_error_response("api_error", &e.to_string(), None))
        }
```

The user-visible divergence point, `src/expressions.rs:427-456` (inside `process_with_cache_groups`):

```rust
        let group_results = if prepared_messages.len() > 1 {
            ...
            run_async(async move {
                crate::utils::fetch_with_cache_warming(          // line 434 — errors ALWAYS JSON
                    &msgs, provider, &m,
                    schema.as_deref(), model_name.as_deref()
                ).await
            })
        } else {
            ...
            run_async(async move {
                if schema.is_some() {
                    crate::utils::fetch_data_message_arrays_with_provider_and_schema(   // line 448 — JSON
                        &msgs, provider, &m,
                        schema.as_deref(), model_name.as_deref()
                    ).await
                } else {
                    crate::utils::fetch_data_message_arrays_with_provider(&msgs, provider, &m).await   // line 453 — NULL
                }
            })
        };
```

The canonical error shape (do NOT change it), `src/model_client/mod.rs:335-349`:

```rust
pub fn create_error_response(error_type: &str, details: &str, raw: Option<&str>) -> String {
    let error_obj = if let Some(raw_content) = raw {
        serde_json::json!({
            "_error": error_type,
            "_details": details,
            "_raw": raw_content
        })
    } else {
        serde_json::json!({
            "_error": error_type,
            "_details": details
        })
    };
    serde_json::to_string(&error_obj).unwrap_or_else(|_| format!(r#"{{"_error": "{}"}}"#, error_type))
}
```

The live test asserting the old contract, `tests/model_client_tests.rs:284-305` (note: no `is_provider_configured` skip — it runs even without keys):

```rust
#[tokio::test]
async fn test_error_handling_invalid_api_key() {
    let _net = network_guard();
    println!("\n🧪 Testing error handling with invalid API key");

    // Temporarily set an invalid API key
    env::set_var("OPENAI_API_KEY", "invalid-key-12345");

    let client = create_client(Provider::OpenAI, "gpt-4o-mini");
    let messages = vec!["Hello".to_string()];

    let results = fetch_data_generic(&*client, &messages).await;

    // Should return None for invalid API key
    assert_eq!(results.len(), 1);
    assert!(results[0].is_none(), "Should return None for invalid API key");

    println!("✓ Error handling works correctly");

    // Clean up
    env::remove_var("OPENAI_API_KEY");
}
```

### Why "always JSON" is the right direction (verified, not assumed)

- The Python structured-output layer already reserves `_error`/`_details`/`_raw` struct fields on every decoded schema (`polar_llama/__init__.py:125-129`).
- The local MLX backend deliberately mirrors `create_error_response` and NEVER emits null on failure (`polar_llama/local/engine.py:58-73`, `polar_llama/local/prefix_cache.py:88-92`) — "Per-row failures return error JSON ... never a batch abort" (`docs/local_mlx_gate_decision.md:337-338`). Standardizing the Rust cloud path on JSON restores parity with the local path.
- Verified during planning: `grep -rn "is_null\|drop_nulls\|fill_null" polar_llama/` returns **no matches** — no Python code in this package branches on null inference results, so nothing in-repo breaks.
- `ModelClientError`'s `Display` (mod.rs:100-109) renders HTTP failures as `HTTP Error {code}: {body}` — both `run_one` and `fetch_with_cache_warming` feed `e.to_string()` into `create_error_response`, so once the flag is gone, all paths produce byte-identical strings for the same failure.

### Constraints

- This is a **public behavior change**: `inference` / `inference_async` / `inference_messages` without `response_format` previously returned null cells for API errors. It must be called out in `CHANGELOG.md` under `[Unreleased]`.
- Keep the return types `Vec<Option<String>>` everywhere — `None` is still used for null *input* rows (mapped at `src/expressions.rs:191`, `:230`, `:386`), just never for failures.
- Do not change any `pub fn` signature. Only the private `run_one`/`run_batch` lose a parameter.
- Documented tradeoffs you must not disturb: pyo3 pinned <0.29 (`.cargo/audit.toml`), reqwest rustls-native-roots/ring choices (`Cargo.toml` comments).
- Commit style: sentence-case imperative summaries, e.g. `Refresh cargo-audit ignore rationale for the pyo3 CVEs` (from `git log`).

## Commands you will need

Run from `/Users/daviddrummond/SideProjects/polar-llama` (all verified in this repo; none need live API keys):

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile lib + tests | `cargo check --tests` | exit 0 |
| Format | `cargo fmt --all` then `cargo fmt --all -- --check` | exit 0, no output |
| Lint (repo standard) | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0 |
| Rust unit tests | `cargo test --lib` | `test result: ok.` |
| Contract tests, keyless | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` | `test result: ok.`, 0 failed |
| Rebuild Python ext | `source .venv/bin/activate && maturin develop` | exit 0 (a `.venv` with maturin exists; if missing: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt maturin`) |
| Python tests | `source .venv/bin/activate && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | 0 failed (live-key tests self-skip) |
| Flag eradication check | `grep -rn "errors_as_json" src/` | no output, exit 1 |

## Scope

**In scope** (the only files you may modify):

- `src/model_client/mod.rs` — remove the `errors_as_json` flag; always emit error-JSON.
- `tests/provider_contract_tests.rs` — update the null assertion in `batch_alignment_mixed_success_failure`; add the three-path equality test.
- `tests/model_client_tests.rs` — update `test_error_handling_invalid_api_key` (lines 284–305) only.
- `CHANGELOG.md` — one `[Unreleased]` entry.
- `plans/README.md` — status row only, if the file exists.

**Out of scope** (do NOT touch, even though they look related):

- `src/utils.rs` and `src/expressions.rs` — no change needed: `fetch_with_cache_warming` already always emits JSON, and the expression routing converges once the flag is gone. If you believe an edit there is required, that is a STOP condition, not a judgment call.
- `create_error_response` and the `{"_error", "_details", "_raw"}` shape — the Python local backend and struct-schema layer mirror it; changing the shape breaks parity.
- Retry / rate-limit behavior — that is plans/008-retry-rate-limit.md.
- Embeddings error handling (`embed_one` at mod.rs:488-499 still returns `None` on failure) — that is plans/005-embeddings-fail-loudly.md's territory.
- All other tests in `tests/model_client_tests.rs`, all Python test files, `polar_llama/` Python sources, README/docs, `.github/workflows/*`.

## Git workflow

- Branch: `advisor/007-unify-error-encoding` (branched from `main` after plans/003's work has landed there — see STOP conditions).
- One commit per step or logical unit; sentence-case imperative summaries, e.g. `Always encode request failures as error JSON` and `Assert uniform error encoding across cache paths`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Pre-flight checks

1. Confirm the dependency landed: `test -f tests/provider_contract_tests.rs && grep -n "batch_alignment_mixed_success_failure" tests/provider_contract_tests.rs` → at least one match. If the file or test is missing, STOP (plans/003 has not been executed).
2. Confirm no in-repo null-dependence: `grep -rn "is_null\|drop_nulls\|fill_null" polar_llama/` → no output (exit 1). If any match handles inference *results* (not inputs), STOP.
3. Run the drift check from the header.

**Verify**: both greps behave as stated above.

### Step 1: Remove the flag from `run_one`, `run_batch`, and the four wrappers

In `src/model_client/mod.rs`:

1. Replace the doc comment at lines 362–366 and drop the `errors_as_json` parameter from `run_one`, making the error arm unconditional:

```rust
/// Core batch runner: sends all requests concurrently (bounded by
/// `max_concurrency`) over the shared HTTP client, preserving input order.
///
/// Request errors are always surfaced as structured error JSON objects
/// (see `create_error_response`), so this never produces a `None` cell —
/// a null output cell always means the input row was null.
async fn run_one<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[Message],
    schema: Option<&str>,
    model_name: Option<&str>,
    schema_check: &SchemaCheck,
) -> Option<String> {
```

with the `Err` arm becoming:

```rust
        Err(e) => {
            eprintln!("Error fetching from {}: {}", client.provider_name(), e);
            Some(create_error_response("api_error", &e.to_string(), None))
        }
```

(The `Ok`/validation arm is unchanged.)

2. Remove `errors_as_json: bool` from `run_batch`'s signature (line 402) and from its `run_one(...)` call (line 410).

3. Update the four wrapper call sites to drop the boolean argument:
   - line 438: `run_batch(client, &message_arrays, None, None).await`
   - line 449: `run_batch(client, &message_arrays, schema, model_name).await`
   - line 457: `run_batch(client, message_arrays, None, None).await`
   - line 467: `run_batch(client, message_arrays, schema, model_name).await`

Do not change any wrapper's `pub` signature, name, or return type. Keep `run_one` returning `Option<String>` (the `Vec<Option<String>>` plumbing and null-input mapping depend on it).

**Verify**:
- `cargo check --tests` → exit 0
- `grep -rn "errors_as_json" src/` → no output, exit 1

### Step 2: Update the plan-003 alignment test to the new contract

In `tests/provider_contract_tests.rs`, find `batch_alignment_mixed_success_failure` (added by plans/003; it mocks the `ROW_BETA` request with a 500 `"boom"` response and currently asserts `assert_eq!(results[1], None);`). Replace that assertion and its comment with:

```rust
    // Row N's failure yields the canonical error JSON at index N without
    // shifting neighbors; nulls are reserved for null input rows.
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].as_deref(), Some("ANSWER_ALPHA"));
    let failed: Value =
        serde_json::from_str(results[1].as_deref().expect("failed row must not be null")).unwrap();
    assert_eq!(failed["_error"], "api_error");
    assert_eq!(failed["_details"], "HTTP Error 500: boom");
    assert_eq!(results[2].as_deref(), Some("ANSWER_GAMMA"));
```

(`Value` is already imported in that file as `serde_json::Value`. If the actual mock 500 body text differs from `"boom"`, match whatever string the existing test mocks.)

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests batch_alignment` → `test result: ok.`, 1 passed.

### Step 3: Add the three-path error-encoding equality test

Append to `tests/provider_contract_tests.rs`. Follow that file's conventions exactly: take `let _env = env_guard();` first, set `OPENAI_BASE_URL` to the mock server URI, remove the var before returning. The three paths under test are the ones `process_with_cache_groups` (src/expressions.rs:427-456) and the non-cached route dispatch to:

- **(a)** cached multi-row group → `polar_llama::utils::fetch_with_cache_warming`
- **(b)** cached single-row group (no schema) → `polar_llama::utils::fetch_data_message_arrays_with_provider`
- **(c)** uncached string path → `polar_llama::model_client::fetch_data_generic`

```rust
/// Same forced failure through all three dispatch paths of
/// process_with_cache_groups (src/expressions.rs) must produce the SAME cell
/// value: the canonical create_error_response JSON — never a null.
#[tokio::test]
async fn error_encoding_identical_across_cache_paths() {
    let _env = env_guard();
    let server = MockServer::start().await;
    std::env::set_var("OPENAI_BASE_URL", server.uri());

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(4) // 2 (warming group) + 1 (single-row group) + 1 (uncached)
        .mount(&server)
        .await;

    // (a) multi-row cache group -> fetch_with_cache_warming (src/utils.rs:148)
    let warmed = polar_llama::utils::fetch_with_cache_warming(
        &[vec![user_msg("ROW_A1")], vec![user_msg("ROW_A2")]],
        Provider::OpenAI,
        "gpt-4o-mini",
        None,
        None,
    )
    .await;

    // (b) single-row cache group, no schema -> fetch_data_message_arrays_with_provider (src/utils.rs:54)
    let single = polar_llama::utils::fetch_data_message_arrays_with_provider(
        &[vec![user_msg("ROW_B1")]],
        Provider::OpenAI,
        "gpt-4o-mini",
    )
    .await;

    // (c) uncached plain-string path -> fetch_data_generic (src/model_client/mod.rs:433)
    let client = create_client(Provider::OpenAI, "gpt-4o-mini");
    let plain = fetch_data_generic(&*client, &["ROW_C1".to_string()]).await;

    // Compare against the canonical encoder itself, not a hardcoded string,
    // so serde_json key ordering can never make this flaky.
    let expected = create_error_response("api_error", "HTTP Error 500: boom", None);
    assert_eq!(warmed.len(), 2);
    assert_eq!(single.len(), 1);
    assert_eq!(plain.len(), 1);
    for cell in warmed.iter().chain(single.iter()).chain(plain.iter()) {
        assert_eq!(cell.as_deref(), Some(expected.as_str()));
    }

    std::env::remove_var("OPENAI_BASE_URL");
}
```

Import notes: the file already imports `create_client`, `fetch_data_generic`, `Provider`, `Message`, and the wiremock items (plans/003 step 3). Add `create_error_response` to the existing `polar_llama::model_client::{...}` import — it is `pub` at mod.rs:335. `polar_llama::utils` is a `pub mod` (src/lib.rs:3), so the `polar_llama::utils::...` calls resolve from an integration test. The failing rows print `Error fetching from ...: HTTP Error 500: boom` lines to stderr — expected noise, not a failure.

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` → `test result: ok.`, 0 failed, output includes `error_encoding_identical_across_cache_paths ... ok`.

### Step 4: Update the live invalid-key test to the new contract

In `tests/model_client_tests.rs`, in `test_error_handling_invalid_api_key` (lines 284–305, quoted in "Current state"), replace only the assertion block:

```rust
    // Failures are data: the cell carries structured error JSON, never null.
    assert_eq!(results.len(), 1);
    let cell = results[0]
        .as_deref()
        .expect("failed request must yield error JSON, not null");
    let parsed: serde_json::Value =
        serde_json::from_str(cell).expect("error cell must be valid JSON");
    assert_eq!(parsed["_error"], "api_error");
```

Leave the rest of the test (env set/remove, `network_guard`, prints) untouched. `serde_json` is a `[dependencies]` crate, so it is available to integration tests without a `Cargo.toml` change. Note the updated test now passes both with network (invalid key → 401 → `api_error` JSON) and without (connection failure → `api_error` JSON) — strictly more robust than before.

**Verify**: `cargo check --tests` → exit 0, and
`grep -c "Should return None for invalid API key" tests/model_client_tests.rs` → `0`.
(Do not require actually running this live test; it needs outbound network. If network is available, `cargo test --test model_client_tests test_error_handling_invalid_api_key` → 1 passed is a bonus check, and it does not need a valid key.)

### Step 5: Add the CHANGELOG entry

In `CHANGELOG.md`, under the existing `## [Unreleased]` heading (line 8 at planning time), add:

```markdown
### Changed
- **Failed requests now always yield structured error JSON** (`{"_error": "api_error", "_details": ...}`) in the output cell, on every code path. Previously, `inference` / `inference_async` / `inference_messages` *without* `response_format` returned a null cell for API errors, while schema-validated and cache-warming paths returned error JSON — so with `cache=True` the encoding of the same failure depended on how many rows shared a system prompt. A null output cell now always and only means the input row was null. If you filtered failures with `.is_null()`, filter on the `_error` key instead (e.g. `.str.contains('"_error"')` or decode the JSON). This also restores parity with the local MLX backend, which has always emitted error JSON.
```

If a `### Changed` subsection already exists under `[Unreleased]`, append the bullet to it instead of duplicating the heading.

**Verify**: `grep -n "always yield structured error JSON" CHANGELOG.md` → one match, on a line number smaller than the `## [0.5.1]` heading's.

### Step 6: Run the full repo gates

```
cargo fmt --all
cargo fmt --all -- --check
RUSTFLAGS="-Dwarnings" cargo clippy --all-features
cargo test --lib
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests
source .venv/bin/activate && maturin develop && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py
```

**Verify**: every command exits 0 / reports 0 failed. (Live-key Python tests self-skip; do not run `cargo test --test model_client_tests` as a gate — it needs network by design.)

## Test plan

- Updated: `batch_alignment_mixed_success_failure` (tests/provider_contract_tests.rs) — the failed middle row now asserts the error-JSON encoding instead of `None`, still proving positional alignment (Step 2).
- Updated: `test_error_handling_invalid_api_key` (tests/model_client_tests.rs) — asserts `_error == "api_error"` instead of `is_none()` (Step 4; live/network test, compile-checked only in the gates).
- New: `error_encoding_identical_across_cache_paths` (tests/provider_contract_tests.rs) — the regression test for this exact bug: one forced 500, three dispatch paths ((a) `fetch_with_cache_warming` multi-row, (b) `fetch_data_message_arrays_with_provider` single-row, (c) `fetch_data_generic` uncached), all four resulting cells byte-equal to `create_error_response("api_error", "HTTP Error 500: boom", None)` (Step 3).
- Structural pattern: the existing wiremock tests in `tests/provider_contract_tests.rs` (env_guard + MockServer + `OPENAI_BASE_URL`), per plans/003.
- Verification: the keyless `cargo test --test provider_contract_tests` command → all pass, 0 failed.

## Done criteria

Machine-checkable. ALL must hold (run from the repo root):

- [ ] `grep -rn "errors_as_json" src/` produces no output (exit 1)
- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `cargo test --lib` exits 0
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` exits 0 with 0 failed, and its output contains `error_encoding_identical_across_cache_paths ... ok`
- [ ] `grep -c "Should return None for invalid API key" tests/model_client_tests.rs` prints `0`, and `cargo check --tests` exits 0
- [ ] `source .venv/bin/activate && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` reports 0 failed (after `maturin develop`)
- [ ] `grep -n "always yield structured error JSON" CHANGELOG.md` shows one match under `[Unreleased]`
- [ ] `git status --porcelain` shows changes only to `src/model_client/mod.rs`, `tests/provider_contract_tests.rs`, `tests/model_client_tests.rs`, `CHANGELOG.md` (and `plans/README.md` if it exists)
- [ ] `plans/README.md` status row updated, if that file exists

## STOP conditions

Stop and report back (do not improvise) if:

- `tests/provider_contract_tests.rs` does not exist or contains no `batch_alignment_mixed_success_failure` test — plans/003-provider-http-mock-tests.md has not landed; this plan's tests need its wiremock harness and env-lock pattern.
- The excerpts in "Current state" don't match the live code (e.g. `run_one` no longer has the `errors_as_json` branch at mod.rs:386-393, `fetch_with_cache_warming` no longer always emits JSON at utils.rs:195/:244, or `test_error_handling_invalid_api_key` has already been rewritten) — the codebase has drifted.
- Step 0's grep finds Python code in `polar_llama/` that branches on null inference *results* (not null inputs) — the null encoding would then be load-bearing downstream, and switching to JSON needs an advisor decision first.
- Removing the flag appears to require changing the signature, name, or return type of any `pub fn` in `src/model_client/mod.rs` or `src/utils.rs`, or any edit to `src/utils.rs` / `src/expressions.rs` at all.
- The three-path equality test fails because the paths produce *different* error strings even after Step 1 (e.g. different `_details` text) — that is a real secondary inconsistency to report, not to paper over by loosening the assertion.
- The Python test suite fails on a test that asserts null-on-error behavior not found during planning.
- Any verification fails twice after a reasonable fix attempt, or `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` fails on pre-existing code unrelated to your changes.

## Maintenance notes

- **This is a public behavior change.** Users who detected failures via `.is_null()` on non-schema inference output will now see error-JSON strings instead. The CHANGELOG entry (Step 5) is the contract; per the repo's stated SemVer adherence (CHANGELOG.md header), the release containing this should bump the minor version (0.6.0), not a patch.
- plans/008-retry-rate-limit.md changes the send path: retry-exhausted failures must keep using `create_error_response("api_error", ...)` so `error_encoding_identical_across_cache_paths` stays green — that test is the guard rail for all future error-path work.
- The local MLX backend mirrors this shape in Python (`polar_llama/local/engine.py:58-73`, `polar_llama/local/prefix_cache.py:88-92`). If `create_error_response` ever gains fields, both mirrors and `polar_llama/__init__.py:125-129` (struct `_error`/`_details`/`_raw` fields) must move in lockstep.
- Reviewer should scrutinize: (a) no `pub fn` signature changed; (b) `src/utils.rs` and `src/expressions.rs` have zero diff; (c) the equality test compares against `create_error_response(...)` output rather than a hardcoded JSON string (key order is serializer-dependent); (d) the updated alignment test still proves positional alignment (index 1 fails, 0 and 2 succeed); (e) `test_error_handling_invalid_api_key` keeps its `network_guard()` and env cleanup.
- Deliberately deferred: embeddings failures still return `None` (`embed_one`, mod.rs:488-499) — plans/005-embeddings-fail-loudly.md; a README "Error Handling" section documenting the uniform contract (README.md:186 only covers the structured-output fields) — small docs follow-up, not gating; collapsing the now nearly-identical `fetch_data_generic*` wrapper pairs — cosmetic, and their signatures are public API.
