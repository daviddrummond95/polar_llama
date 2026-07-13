# Plan 019: Convert task-join and runtime `.expect()` panics into PolarsResult errors at the FFI boundary

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/expressions.rs src/utils.rs src/cost.rs`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/002-ci-verification-gates.md (soft — that plan makes CI run `cargo test`, so the new unit tests added here become gating; this plan is fully executable locally before 002 lands)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

Three `.expect()` calls sit on the plugin hot path in `src/expressions.rs`:
the `run_async` helper re-panics when a spawned tokio task panics (a
`JoinError`), and `execute_tool_calls` has two more expects on semaphore
acquisition and task join. When any of them fires, the panic is caught by the
`catch_unwind` wrapper that pyo3-polars generates around every
`#[polars_expr]` function, and the Python user sees only the literal error
message `PANIC` (plus raw panic text on stderr) instead of a real error. This
plan converts those panics into `PolarsError::ComputeError` values with
descriptive messages that propagate cleanly as Python `ComputeError`
exceptions, and makes tool-call task panics per-row error data instead of
whole-query failures. Happy-path behavior does not change.

**Investigation result (recorded per the finding's step 1, verified
2026-07-06):** `Cargo.lock` pins `pyo3-polars-derive` **0.20.0** (pulled in by
`pyo3-polars` 0.26.0). Its macro expansion wraps the entire expression body in
`std::panic::catch_unwind` — from
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/pyo3-polars-derive-0.20.0/src/lib.rs:149-160`:

```rust
let panic_result = std::panic::catch_unwind(move || {
    let inputs = polars_ffi::version_0::import_series_buffer(e, input_len).unwrap();
    #quote_call
    #quote_process_result
});
if panic_result.is_err() {
    // Set latest to panic;
    ::pyo3_polars::derive::_set_panic();
}
```

and `_set_panic` (in `pyo3-polars-0.26.0/src/derive.rs:32-36`) sets the
last-error message to the literal string `"PANIC"`. So a panic does **not**
unwind across the C ABI and does not abort the Python process. Per the
finding, this plan is therefore the downgraded variant: convert the expects
into clean `PolarsResult` errors for better messages. Step 1 below has the
executor re-verify this so the plan-execution report records it from the
executor's own environment.

## Current state

All excerpts verified against the working tree at commit `afc78da`.

Relevant files:

- `src/expressions.rs` — all `#[polars_expr]` registrations; contains
  `run_async` (lines 17–24), its 8 call sites, and `execute_tool_calls`
  (lines 948–1124) with the two tool-call expects.
- `src/utils.rs` — shared tokio runtime `RT` (lines 9–11); its `.expect()` is
  startup-only and stays, gaining only a comment.
- `src/cost.rs` — tokenizer `LazyLock`s with `.expect()` in the init closure
  (lines 275–281); investigated below, gains only a comment.

### The `run_async` helper — `src/expressions.rs:14-24`

```rust
// Helper function to run async operations in a way that allows true parallelization
// Instead of directly blocking on async work, we spawn it as a task and then block on the handle
// This allows multiple threads to spawn tasks that run concurrently on the runtime's thread pool
fn run_async<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let handle = RT.spawn(future);
    RT.block_on(handle).expect("Task panicked")
}
```

The three-line comment above the function is a documented design rationale —
keep it verbatim; do not redesign the spawn-then-block pattern.

### All 8 `run_async` call sites (from `grep -n "run_async(" src/expressions.rs`)

| Line | Enclosing function | Enclosing return type | Change needed |
|------|--------------------|-----------------------|---------------|
| 215  | `inference_async` | `PolarsResult<Series>` | append `?` |
| 369  | `inference_messages` | `PolarsResult<Series>` | append `?` |
| 380  | `inference_messages` | `PolarsResult<Series>` | append `?` |
| 433  | `process_with_cache_groups` | `Vec<Option<String>>` today — becomes `PolarsResult<Vec<Option<String>>>` | append `?` |
| 446  | `process_with_cache_groups` | (same) | append `?` |
| 619  | `embedding_async` | `PolarsResult<Series>` | append `?` |
| 1040 | `execute_tool_calls` | `PolarsResult<Series>` | `run_async(...)?` before the existing `.map_err(...)?` |
| 1077 | `execute_tool_calls` | `PolarsResult<Series>` | append `?` |

`process_with_cache_groups` (defined at `src/expressions.rs:397`, currently
returning plain `Vec<Option<String>>`) is called from exactly two places, both
inside `PolarsResult<Series>` functions:

- `src/expressions.rs:180` (in `inference_async`): `let api_results = process_with_cache_groups(...);`
- `src/expressions.rs:353` (in `inference_messages`, first branch of the
  `let api_results = if cache_enabled { ... } else if ... }` chain)

so changing its return type to `PolarsResult<Vec<Option<String>>>` composes
with `?` at both call sites. **No `run_async` caller sits in a
non-`PolarsResult` context** — verified at `afc78da`.

The MCP-connect call site today (`src/expressions.rs:1040-1045`) — note the
future's output is itself a `Result<McpHttpClient, String>` (see
`src/mcp.rs:50`: `pub async fn connect(endpoint: &str, timeout: Duration) -> Result<Self, String>`):

```rust
    let client = run_async(async move { crate::mcp::McpHttpClient::connect(&transport, timeout).await })
        .map_err(|e| {
            PolarsError::ComputeError(
                format!("failed to connect to MCP server '{}': {e}", kwargs.transport).into(),
            )
        })?;
```

### The two tool-call expects — `src/expressions.rs:1061-1086`

```rust
                Prepared::Ready(args) => {
                    let client = client.clone();
                    let semaphore = semaphore.clone();
                    handles.push(TaskOrReady::Task(RT.spawn(async move {
                        let _permit = semaphore.acquire().await.expect("semaphore closed");
                        let outcome = client.call_tool(&spec.tool_name, &args).await;
                        (
                            row_idx,
                            call_idx,
                            tool_result_json(&spec, outcome.content, outcome.is_error, outcome.error),
                        )
                    })));
                }
            }
        }

        run_async(async move {
            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                match handle {
                    TaskOrReady::Ready(r) => results.push(r),
                    TaskOrReady::Task(t) => results.push(t.await.expect("tool call task panicked")),
                }
            }
            results
        })
```

The helper enum is defined at `src/expressions.rs:1126-1129`:

```rust
enum TaskOrReady {
    Task(tokio::task::JoinHandle<(usize, usize, serde_json::Value)>),
    Ready((usize, usize, serde_json::Value)),
}
```

`execute_tool_calls`'s documented error contract (doc comment at
`src/expressions.rs:942-947`) is: "Per-call failures are data, not
exceptions; only an unreachable/misconfigured transport raises." The fix
below honors it: a panicking tool-call task becomes a per-call error JSON
object, not a query failure. The existing per-row parse-error shape to imitate
is at `src/expressions.rs:1108-1115`:

```rust
                if let Some(parse_err) = &row_errors[idx] {
                    values.push(serde_json::json!({
                        "tool_name": null,
                        "arguments": null,
                        "content": null,
                        "is_error": true,
                        "_error": parse_err,
                    }));
                }
```

and `tool_result_json` is defined at `src/expressions.rs:924-941` with
signature `fn tool_result_json(spec: &ToolCallSpec, content: Option<String>, is_error: bool, error: Option<String>) -> serde_json::Value`.

### The startup-only expects (keep, comment only)

`src/utils.rs:9-11`:

```rust
/// Global Tokio runtime shared by all blocking entry points.
pub(crate) static RT: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Failed to create Tokio runtime"));
```

`src/cost.rs:274-281`:

```rust
/// Cached tokenizer instances for performance
static CL100K_TOKENIZER: LazyLock<CoreBPE> = LazyLock::new(|| {
    cl100k_base().expect("Failed to load cl100k_base tokenizer")
});

static O200K_TOKENIZER: LazyLock<CoreBPE> = LazyLock::new(|| {
    o200k_base().expect("Failed to load o200k_base tokenizer")
});
```

**Investigation result for cost.rs (finding step: "harden or document"):**
`cl100k_base()` / `o200k_base()` from `tiktoken-rs` 0.12 build tokenizers from
vocabulary data compiled into the binary — there is no I/O and no realistic
runtime failure mode. The `LazyLock` initializes on first
`count_tokens` call (`src/cost.rs:314-329`), which the existing unit tests
`cost::tests::test_token_counting` and `test_gpt4o_tokenizer` already
exercise on every `cargo test --lib` run. And if it ever did panic, the
pyo3-polars `catch_unwind` wrapper contains it. Decision: **document, do not
harden** — a comment only (Step 4).

### Conventions

- Rust error handling: expression-level failures are
  `PolarsError::ComputeError(format!(...).into())` — exemplar at
  `src/expressions.rs:1041-1044` (quoted above). Match it.
- Unit tests live in a `#[cfg(test)] mod tests { use super::*; ... }` block
  at the bottom of the source file — exemplar: `src/cost.rs:344-369`.
- `Cargo.toml` has `crate-type = ["cdylib", "rlib"]`, so `cargo test --lib`
  works (15 unit tests pass today).

## Commands you will need

Run everything from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Fast compile check | `cargo check` | `Finished` line, exit 0 |
| Rust unit tests | `cargo test --lib` | today: `15 passed; 0 failed`; after Step 5: `17 passed; 0 failed` |
| Format check | `cargo fmt --all -- --check` | no output, exit 0 |
| Lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | `Finished` line, exit 0 (verified clean at `afc78da`) |
| Rebuild Python module | `source .venv/bin/activate && uvx maturin develop` | ends with an `Installed polar-llama-0.5.1` (or similar) line |
| Python tests | `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | same pass/skip/fail counts as the baseline you record in Step 0 (200 tests collected at `afc78da`) |

Notes:

- The repo's `.venv` is uv-managed and has **no pip module**; `maturin` is not
  installed in it. Use `uvx maturin develop` with the venv activated (`uv` is
  at `~/.local/bin/uv`). If `uvx` is unavailable, fall back to
  `uv pip install maturin --python .venv/bin/python` then
  `source .venv/bin/activate && maturin develop`.
- **Never** set or require live API keys (`OPENAI_API_KEY` etc.). The pytest
  command above is the CI-safe subset. `tests/model_client_tests.rs` (Rust
  integration tests) needs live keys — do not run it; `cargo test --lib`
  deliberately excludes it.

## Scope

**In scope** (the only files you should modify):

- `src/expressions.rs` — `run_async` signature, its 8 call sites,
  `process_with_cache_groups` signature, the two tool-call expects, the
  `TaskOrReady` enum, plus a new `#[cfg(test)]` module.
- `src/utils.rs` — comment only (on the `RT` static).
- `src/cost.rs` — comment only (on the tokenizer statics).
- `plans/README.md` — status row update at the end (unless your dispatcher
  said they maintain the index).

**Out of scope** (do NOT touch, even though they look related):

- `pyo3-polars` / `pyo3-polars-derive` — vendored dependencies; also do not
  bump `pyo3` (pinned `<0.29` by pyo3-polars; rationale in
  `.cargo/audit.toml`).
- `Cargo.toml` — no panic-strategy change (leave `panic = "unwind"` default),
  no dependency changes.
- The spawn-then-block pattern in `run_async` — the rationale comment at
  `src/expressions.rs:14-16` stays valid and stays verbatim.
- `src/utils.rs` beyond the one comment — in particular
  `fetch_api_response_sync_with_provider`'s direct `RT.block_on` (line 91) is
  not a spawn and has no JoinError to map.
- `tests/model_client_tests.rs`, the Python layer (`polar_llama/`), and all
  other `src/` modules.

## Git workflow

- Branch: `advisor/019-ffi-panic-hardening` (branched from `main`)
- Commit per step or per logical unit; message style is sentence-case
  imperative, e.g. from `git log`: "Refresh cargo-audit ignore rationale for
  the pyo3 CVEs"
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Record the Python test baseline

Before touching any code, run:

```
.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py 2>&1 | tail -3
```

Save the final summary line (counts of passed/skipped/failed) — the done
criterion is that these counts are unchanged after the fix. (Some tests may
skip without API keys; that is fine as long as the profile is identical
before and after.)

**Verify**: the command completes and prints a summary line with `200`
collected tests (a small drift in total count is acceptable if the drift
check passed; a large one is a STOP).

### Step 1: Re-verify the catch_unwind guard (record in your report)

```
grep -n "catch_unwind" ~/.cargo/registry/src/*/pyo3-polars-derive-0.20.0/src/lib.rs
```

**Verify**: two matches (around lines 149 and 213), confirming the
`#[polars_expr]` macro wraps calls in `std::panic::catch_unwind`. Record
"catch_unwind confirmed at derive 0.20.0" in your plan-execution report. If
the directory does not exist, run `cargo fetch` first. If grep finds **no**
`catch_unwind`, do not stop — the fix below is identical and strictly more
important; record that instead.

### Step 2: Make `run_async` return `PolarsResult<T>` and update all 8 call sites

In `src/expressions.rs`, replace the body of `run_async` (keep the 3-line
rationale comment above it verbatim):

```rust
fn run_async<F, T>(future: F) -> PolarsResult<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let handle = RT.spawn(future);
    RT.block_on(handle).map_err(|e| {
        PolarsError::ComputeError(format!("polar-llama task panicked: {e}").into())
    })
}
```

Then update the call sites per the table in "Current state":

1. Line 215 (`inference_async`), line 619 (`embedding_async`), lines 369 and
   380 (`inference_messages`): append `?` to the closing `)` of each
   `run_async(...)` expression, e.g. line 215's `});` becomes `})?;` and the
   branch expressions at 369/380 become `run_async(async move { ... })?`.
2. Change `process_with_cache_groups` (line 397) to return
   `PolarsResult<Vec<Option<String>>>`:
   - signature: `) -> PolarsResult<Vec<Option<String>>> {`
   - append `?` to the two `run_async(...)` expressions at lines 433 and 446
     (the `if`/`else` branch values become `Vec<Option<String>>` again)
   - wrap the final expression (line 467) in `Ok(...)`:
     `Ok(all_results.into_iter().map(|(_, r)| r).collect())`
3. Append `?` at its two call sites: line 180 (`inference_async`) —
   `let api_results = process_with_cache_groups(...)?;` — and line 353
   (`inference_messages` branch) — `process_with_cache_groups(...)?`.
4. Line 1040 (`execute_tool_calls`): the future's output is itself a
   `Result`, so unwrap the join layer first, keeping the existing
   `.map_err(...)` for the connect error:

   ```rust
   let client = run_async(async move { crate::mcp::McpHttpClient::connect(&transport, timeout).await })?
       .map_err(|e| {
           PolarsError::ComputeError(
               format!("failed to connect to MCP server '{}': {e}", kwargs.transport).into(),
           )
       })?;
   ```
5. Line 1077: append `?` to the `run_async(...)` that collects tool-call
   outcomes (final form shown in Step 3).

**Verify**: `cargo check` → exit 0. Then
`grep -c "run_async(" src/expressions.rs` → `8` (definition line 17 does not
contain `run_async(` with this exact pattern; if you get 9, one call site was
missed — recount against the table).

### Step 3: Convert the two tool-call expects into per-call error data

Still in `src/expressions.rs`. The join handle currently carries the row/call
indices inside the task, so a `JoinError` would lose them; move the indices
out of the task.

1. Change the `TaskOrReady` enum (lines 1126-1129) to:

   ```rust
   enum TaskOrReady {
       Task(usize, usize, tokio::task::JoinHandle<serde_json::Value>),
       Ready((usize, usize, serde_json::Value)),
   }
   ```

2. Replace the spawn arm (lines 1061-1073). The semaphore is created locally
   and never closed, so `acquire()` failing is unreachable in practice — but
   convert it to the documented per-call error shape via `tool_result_json`
   (`spec` is owned by the task, so it is available here):

   ```rust
   Prepared::Ready(args) => {
       let client = client.clone();
       let semaphore = semaphore.clone();
       handles.push(TaskOrReady::Task(
           row_idx,
           call_idx,
           RT.spawn(async move {
               let _permit = match semaphore.acquire().await {
                   Ok(permit) => permit,
                   Err(e) => {
                       return tool_result_json(
                           &spec,
                           None,
                           true,
                           Some(format!("internal error: semaphore closed: {e}")),
                       );
                   }
               };
               let outcome = client.call_tool(&spec.tool_name, &args).await;
               tool_result_json(&spec, outcome.content, outcome.is_error, outcome.error)
           }),
       ));
   }
   ```

3. Replace the join loop (lines 1077-1086). On a `JoinError` the `spec` is
   gone (it was moved into the panicked task), so emit the same
   null-`tool_name` error object the row-parse-error path already uses
   (`src/expressions.rs:1108-1115`):

   ```rust
   run_async(async move {
       let mut results = Vec::with_capacity(handles.len());
       for handle in handles {
           match handle {
               TaskOrReady::Ready(r) => results.push(r),
               TaskOrReady::Task(row_idx, call_idx, t) => {
                   let value = t.await.unwrap_or_else(|e| {
                       serde_json::json!({
                           "tool_name": null,
                           "arguments": null,
                           "content": null,
                           "is_error": true,
                           "_error": format!("tool call task panicked: {e}"),
                       })
                   });
                   results.push((row_idx, call_idx, value));
               }
           }
       }
       results
   })?
   ```

   (The trailing `?` is the Step 2 item 5 change — the block is the value of
   `let outcomes: Vec<(usize, usize, serde_json::Value)> = { ... };`.)

**Verify**: `cargo check` → exit 0, and
`grep -n "expect(" src/expressions.rs` → no output (exit code 1).

### Step 4: Document the two intentionally-kept startup expects

1. `src/utils.rs` — above the `RT` static (lines 9-11), extend the doc
   comment, e.g.:

   ```rust
   /// Global Tokio runtime shared by all blocking entry points.
   /// The `.expect()` here is intentional: it runs once at first use, and a
   /// process that cannot create a tokio runtime cannot do anything useful.
   /// Per-task panics are handled in `expressions::run_async`, which maps
   /// `JoinError` to `PolarsError::ComputeError` instead of re-panicking.
   ```

2. `src/cost.rs` — above `CL100K_TOKENIZER` (line 275), extend the comment:

   ```rust
   /// Cached tokenizer instances for performance.
   /// The `.expect()`s are init-once and load vocabulary data compiled into
   /// the binary (no I/O), so they cannot realistically fail at runtime;
   /// `cargo test --lib` exercises both on every run. A hypothetical panic
   /// is contained by the catch_unwind wrapper pyo3-polars generates around
   /// every #[polars_expr] entry point.
   ```

Do not change any executable code in these two files.

**Verify**: `git diff --stat src/utils.rs src/cost.rs` → each file shows only
a few changed lines; `cargo check` → exit 0.

### Step 5: Add the panic-mapping unit tests

At the bottom of `src/expressions.rs`, add (model the structure after
`src/cost.rs:344-369`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_async_returns_value_on_success() {
        let result = run_async(async { 42usize });
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn run_async_maps_task_panic_to_polars_error() {
        let result: PolarsResult<()> = run_async(async {
            panic!("deliberate test panic");
        });
        let err = result.expect_err("panicking task must yield Err, not abort");
        assert!(
            err.to_string().contains("polar-llama task panicked"),
            "unexpected error text: {err}"
        );
    }
}
```

Note: the panicking task will print a `thread 'tokio-runtime-worker'
panicked` message to test stderr — that is expected and does not fail the
test. (`expect_err(` does not match the done-criteria grep pattern
`expect(` — the substring differs — so the test code keeps
`grep -n "expect(" src/expressions.rs` returning nothing.)

**Verify**: `cargo test --lib` → `17 passed; 0 failed` (15 pre-existing + 2
new, including `expressions::tests::run_async_maps_task_panic_to_polars_error`).

### Step 6: Full local gate

```
cargo fmt --all
cargo fmt --all -- --check
RUSTFLAGS="-Dwarnings" cargo clippy --all-features
cargo test --lib
source .venv/bin/activate && uvx maturin develop
.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py
```

**Verify**: fmt-check silent/exit 0; clippy exits 0; `17 passed`; maturin
prints an `Installed` line; pytest summary counts equal the Step 0 baseline.

## Test plan

- New tests (Step 5) in `src/expressions.rs` `#[cfg(test)] mod tests`:
  - `run_async_returns_value_on_success` — happy path: the wrapper still
    yields the future's value.
  - `run_async_maps_task_panic_to_polars_error` — the regression this plan
    exists for: a panicking spawned future produces
    `Err(PolarsError::ComputeError("polar-llama task panicked: ..."))`
    instead of re-panicking (and instead of aborting the test process).
- Structural pattern: `src/cost.rs:344-369` (`mod tests` + `use super::*`).
- The tool-call join-error branch is not directly unit-tested (constructing a
  panicking MCP tool call requires a live transport); it is covered
  indirectly by the type change (`JoinHandle<serde_json::Value>` forces the
  `unwrap_or_else` handling) and by the unchanged pytest tool-use suite
  (`tests/test_tool_use.py`) confirming happy-path behavior is a no-op.
- Verification: `cargo test --lib` → 17 passed; pytest command → baseline
  counts.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "expect(" src/expressions.rs` returns **no output** (exit
      code 1): the three original sites (old lines 23, 1065, 1082) are gone.
      (`expect_err(` in the Step 5 test does not match this pattern.)
- [ ] `grep -n "run_async(" src/expressions.rs` shows every call site
      immediately followed by error propagation (`?`) — 8 call sites, none
      returning a bare `T`.
- [ ] `cargo test --lib` exits 0 with `17 passed; 0 failed`, including
      `run_async_maps_task_panic_to_polars_error`.
- [ ] `cargo fmt --all -- --check` exits 0.
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0.
- [ ] `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
      reports the same passed/skipped/failed counts as the Step 0 baseline.
- [ ] `git status --porcelain` shows changes only in `src/expressions.rs`,
      `src/utils.rs`, `src/cost.rs` (and `plans/README.md` if you maintain
      the index).
- [ ] `plans/README.md` status row updated (unless dispatcher maintains it).

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows changes to `src/expressions.rs`, `src/utils.rs`, or
  `src/cost.rs` since `afc78da` and the "Current state" excerpts no longer
  match the live code.
- `grep -n "run_async(" src/expressions.rs` finds a call site **not** in the
  8-row table above, or any call site turns out to be in a function that
  cannot return `PolarsResult` (report which function).
- After Step 2, `cargo check` fails twice with type errors you cannot resolve
  by exactly following the table (a call-site context changed).
- The Step 5 panic test **aborts the process** instead of returning `Err`
  (would mean panics are not unwinding — e.g. a `panic = "abort"` profile
  crept in; that invalidates this plan's approach).
- The Step 6 pytest counts differ from the Step 0 baseline in any direction
  (this plan must be a behavioral no-op on the happy path).
- Fixing anything appears to require touching `Cargo.toml`, `polar_llama/`,
  or any `src/` file other than the three in scope.

## Maintenance notes

- Any **new** `#[polars_expr]` function that does async work must call
  `run_async(...)?` — the `PolarsResult` return type now enforces this at
  compile time. Do not reintroduce `.expect()` on `JoinHandle`s.
- Reviewer focus: (a) every `run_async` call site got `?` — a missed one is a
  compile error, but check no call was rewritten to `.unwrap()` instead;
  (b) the `execute_tool_calls` join loop preserves `(row_idx, call_idx)`
  ordering semantics — indices now travel in the `TaskOrReady::Task` variant
  rather than the task's return tuple; (c) the JoinError JSON object matches
  the parse-error shape at `src/expressions.rs:1108-1115` so downstream
  consumers of `_error` see one schema.
- Deferred (intentionally out of scope): unifying the error-JSON shapes
  themselves is plans/007-unify-error-encoding.md territory; retry behavior
  on transient failures is plans/008-retry-rate-limit.md; a timeout around
  tool-call tasks is plans/009-tool-executor-timeout.md.
- The pyo3-polars `catch_unwind` guard (derive 0.20.0) remains the backstop
  for any panic this plan did not convert. If the project ever bumps
  `pyo3-polars`, re-verify the guard still exists in the new derive version
  (Step 1's grep, adjusted for version).
