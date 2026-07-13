# Plan 009: Make the Python-executor tool-call timeout actually bound wall-clock (no hang on stuck executors)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- polar_llama/tools.py tests/test_tool_use.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

`execute_tool_calls(..., executor=...)` documents `timeout_s` as a "per-call
timeout", but the Python-executor path cannot actually enforce it. A hung
executor thread (e.g. a tool call stuck on a dead database connection) is
never cancelled: even after `future.result(timeout=timeout_s)` raises
`TimeoutError`, leaving the `with ThreadPoolExecutor(...)` block calls
`shutdown(wait=True)`, which blocks until the stuck call returns — so one hung
tool call hangs the entire Polars expression forever. Separately, the timeouts
compound: each `future.result(timeout=timeout_s)` wait starts only after the
previous future resolved, so N stuck calls take up to N x `timeout_s` before
even reaching the (also-blocking) shutdown. After this plan, the
Python-executor path returns within roughly `timeout_s` no matter what the
executor does, with timed-out calls reported as error *data* in the results
column — consistent with the module's "failures are data, not exceptions"
convention.

## Current state

Relevant files:

- `polar_llama/tools.py` — tool-use module. `execute_tool_calls` (lines
  405–498) is the public API; the buggy code is in `_execute_batch_python`
  (lines 531–570).
- `tests/test_tool_use.py` — keyless tests for tool use; the Python-executor
  tests start at line 321 (`test_execute_tool_calls_python_executor`).
- `src/expressions.rs` / `src/mcp.rs` — the Rust/MCP transport path.
  **Verified during planning, no change needed**: `src/mcp.rs:50-54` builds the
  `reqwest::Client` with `.timeout(timeout)`, so every `tools/call` HTTP
  request on the Rust path is already bounded per-request by `timeout_s`. Do
  not touch Rust code in this plan.

### The buggy code, as it exists at `afc78da`

`polar_llama/tools.py:20` (the only `concurrent.futures` import today):

```python
from concurrent.futures import ThreadPoolExecutor
```

`polar_llama/tools.py:549-562` (inside `_execute_batch_python`):

```python
    results: Dict[tuple, Dict[str, Any]] = {}
    if flat:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {
                pool.submit(_run_one_python_call, executor, call, tool_schemas): (row_idx, call_idx)
                for row_idx, call_idx, call in flat
            }
            for future, key in futures.items():
                row_idx, call_idx = key
                call = rows[row_idx][call_idx]
                try:
                    results[key] = future.result(timeout=timeout_s)
                except Exception as e:  # includes TimeoutError
                    results[key] = _tool_result(call, None, True, f"{type(e).__name__}: {e}")
```

Defect 1 (liveness): the `with` block's implicit `pool.shutdown(wait=True)`
joins all worker threads, so a hung `_run_one_python_call` blocks forever.
Defect 2 (compounding): the sequential `.result(timeout=timeout_s)` calls give
each future a fresh `timeout_s` budget after the previous one resolves.

Reassembly immediately after (`polar_llama/tools.py:564-570`) — this must
keep working unchanged; it requires `results` to contain an entry for every
`(row_idx, call_idx)` key:

```python
    out: List[Optional[List[Dict[str, Any]]]] = []
    for row_idx, row in enumerate(rows):
        if row is None:
            out.append(None)
        else:
            out.append([results[(row_idx, call_idx)] for call_idx in range(len(row))])
    return pl.Series(series.name, out, dtype=TOOL_RESULT_DTYPE)
```

The docstring line being contradicted (`polar_llama/tools.py:451-452`, in
`execute_tool_calls`):

```python
    timeout_s : int
        Per-call timeout in seconds (default 30).
```

Error-result helper the fix must reuse (`polar_llama/tools.py:642-657`):
`_tool_result(call, content, is_error, error)` returns the
`{tool_name, arguments, content, is_error, _error}` dict. Timed-out calls
must be filled with `_tool_result(call, None, True, "TimeoutError: ...")`.

### Chosen timeout semantic (implement exactly this)

**Deadline-based, measured from batch submission.** All futures are submitted
at once, so a per-call timeout measured from submission is identical for every
call: a single deadline of `time.monotonic() + timeout_s` taken right after
submission. Any future not finished by that deadline is reported as a
`TimeoutError` result. This bounds the whole Python-executor batch at roughly
`timeout_s` wall-clock. Trade-offs to document in the docstring (Step 2):

- When there are more calls than `concurrency` and earlier calls are slow,
  calls still queued at the deadline are also reported as timed out (they are
  cancelled before starting).
- Threads already running a hung call are *abandoned*, not killed: the
  expression returns promptly, but the thread may keep running in the
  background and its eventual result is discarded. (Python threads cannot be
  forcibly cancelled; this is the standard library's limit.)

### Conventions that apply

- Failures are data: per-call failures go into `is_error` / `_error` fields,
  never exceptions. See the module docstring comment at
  `polar_llama/tools.py:33-35` and the exemplar handling in
  `_run_one_python_call` (`polar_llama/tools.py:593-596`).
- `shutdown(wait=False, cancel_futures=True)` requires Python >= 3.9.
  Verified: `pyproject.toml:16` says `requires-python = ">=3.9"`, and
  `cancel_futures` was added to `Executor.shutdown` in exactly 3.9. Safe.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Activate env | `source /Users/daviddrummond/SideProjects/polar-llama/.venv/bin/activate` | prompt shows `(.venv)` |
| Sanity import | `python -c "import polar_llama; print('ok')"` | prints `ok` |
| (Only if import fails) rebuild | `maturin develop` (inside the venv, repo root) | exit 0 |
| Tool-use tests | `pytest tests/test_tool_use.py -v` | all pass (14 pass today at `afc78da`; 16+ after this plan) |
| Full keyless suite | `pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | exit 0, no new failures vs. baseline |

Run all commands from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.
Note: the working tree at `afc78da` has a few untracked `test_*.py` files at
the repo root (`test_cache_messages.py` etc.) — leave them alone; the suite
command above only collects `tests/`.

## Scope

**In scope** (the only files you may modify):
- `polar_llama/tools.py` — `_execute_batch_python`, the `timeout_s` docstring
  in `execute_tool_calls`, and the `concurrent.futures` import line.
- `tests/test_tool_use.py` — new timeout tests.

**Out of scope** (do NOT touch, even though they look related):
- `src/expressions.rs`, `src/mcp.rs` — the Rust/MCP path already has a
  per-request timeout via the reqwest client (`src/mcp.rs:50-54`); verified,
  no change needed.
- The MCP HTTP path in `polar_llama/tools.py` (`_McpHttpSession`,
  `_McpStdioSession`, `mcp_tools`) — different code path, has its own timeout.
- Default values of `concurrency` (32) and `timeout_s` (30) — behavior
  defaults are unchanged.
- `plans/README.md` — the advisor maintains the index; do not create or edit
  it unless you were told you own it.

## Git workflow

- Branch: `advisor/009-tool-executor-timeout` (create from `main`).
- Commit style: sentence-case imperative summary, matching `git log` (e.g.
  "Refresh cargo-audit ignore rationale for the pyo3 CVEs"). Suggested:
  "Bound execute_tool_calls Python-executor wall-clock with a shared deadline".
- One commit for the fix + tests is fine; do NOT push or open a PR unless the
  operator instructed it.

## Steps

### Step 1: Replace the sequential per-future waits with a shared deadline

In `polar_llama/tools.py`:

1. Change the import at line 20 to:

```python
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
```

2. Add `import time` to the stdlib import block near the top of the file
   (alphabetical order with the existing `import json` / `import shlex` /
   `import subprocess` group).

3. Replace the `if flat:` block shown in "Current state"
   (`polar_llama/tools.py:550-562`) with a deadline-based wait. Target shape:

```python
    results: Dict[tuple, Dict[str, Any]] = {}
    if flat:
        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        try:
            futures = {
                pool.submit(_run_one_python_call, executor, call, tool_schemas): (row_idx, call_idx)
                for row_idx, call_idx, call in flat
            }
            # One shared deadline for the whole batch: every call was
            # submitted at the same instant, so "per-call timeout from
            # submission" and "batch deadline" are the same thing.
            done, not_done = _futures_wait(futures, timeout=timeout_s)
            for future in done:
                key = futures[future]
                row_idx, call_idx = key
                call = rows[row_idx][call_idx]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = _tool_result(call, None, True, f"{type(e).__name__}: {e}")
            for future in not_done:
                key = futures[future]
                row_idx, call_idx = key
                call = rows[row_idx][call_idx]
                results[key] = _tool_result(
                    call,
                    None,
                    True,
                    f"TimeoutError: tool call did not finish within {timeout_s}s",
                )
        finally:
            # Cancel queued calls; abandon (do not join) already-running
            # threads so a hung executor cannot hang the expression. Hung
            # threads may keep running in the background; their results are
            # discarded.
            pool.shutdown(wait=False, cancel_futures=True)
```

Notes on the target shape:
- Keep the `futures` dict keyed future -> `(row_idx, call_idx)` exactly as
  today; the reassembly loop below it (`out.append([results[(row_idx, call_idx)] ...])`)
  must not change, and every key must have a `results` entry (done futures via
  `.result()`, not-done futures via the TimeoutError `_tool_result`).
- `future.result()` on a done future does not block; keep the
  `except Exception` because `_run_one_python_call` catching everything is a
  convention, not a guarantee. `CancelledError` cannot surface here: nothing
  cancels futures until the `finally` block, which runs after the `done` set
  has been fully processed, so every future in `done` either has a result or
  a stored exception. (Do NOT widen to `except BaseException` — since Python
  3.8 `concurrent.futures.CancelledError` is a `BaseException`, but as noted
  it is unreachable in this shape, and swallowing `KeyboardInterrupt` would
  be wrong.)
- The `with` statement must go away — it is precisely the
  `shutdown(wait=True)` that causes the hang. Use the explicit
  `try/finally: pool.shutdown(wait=False, cancel_futures=True)` shown.
- The error string must start with `"TimeoutError: "` — the tests grep for it
  and it matches the old `f"{type(e).__name__}: {e}"` format users may parse.

**Verify**: `pytest tests/test_tool_use.py -v` → all 14 existing tests still
pass (no new tests yet).

### Step 2: Update the `timeout_s` docstring in `execute_tool_calls`

Replace `polar_llama/tools.py:451-452`:

```python
    timeout_s : int
        Per-call timeout in seconds (default 30).
```

with wording that documents the real semantics of both paths, e.g.:

```python
    timeout_s : int
        Timeout in seconds (default 30). On the MCP ``transport`` path this
        bounds each HTTP request. On the Python ``executor`` path it is a
        wall-clock deadline for the whole batch, measured from submission:
        calls not finished by the deadline (including calls still queued
        behind slow ones when there are more calls than ``concurrency``)
        are reported as ``TimeoutError`` error data, queued calls are
        cancelled, and already-running executor threads are abandoned —
        they may keep running in the background but their results are
        discarded (Python threads cannot be force-killed).
```

**Verify**: `python -c "import polar_llama; help(polar_llama.execute_tool_calls)" | grep -A3 timeout_s` → shows the new wording.

### Step 3: Add timeout tests

Add the following tests to `tests/test_tool_use.py`, in the
"Phase 1: execution — Python executor escape hatch" section (after
`test_null_rows_stay_null`, line 359-363). Model them on
`test_execute_tool_calls_python_executor` (line 321). Add `import time` to the
test file's imports (top of file, with `import json` / `import threading`).

```python
def test_execute_tool_calls_python_executor_timeout_is_data():
    """A hung executor call becomes TimeoutError data; fast calls still succeed."""
    def executor(tool_name, arguments):
        if arguments.get("query") == "hang":
            time.sleep(5)
        return "ok"

    df_input = pl.DataFrame({
        "calls": [
            json.dumps([
                {"tool_name": "t", "arguments": json.dumps({"query": "hang"})},
                {"tool_name": "t", "arguments": json.dumps({"query": "fast"})},
            ]),
            json.dumps([
                {"tool_name": "t", "arguments": json.dumps({"query": "also fast"})},
            ]),
        ],
    })

    start = time.monotonic()
    df = df_input.with_columns(
        results=execute_tool_calls(pl.col("calls"), executor=executor, timeout_s=1)
    )
    elapsed = time.monotonic() - start

    # Old code blocked in shutdown(wait=True) until the sleep(5) finished
    # (>= 5s). The deadline-based wait returns in ~1s; 3s is a generous
    # CI-safe bound that still discriminates.
    assert elapsed < 3.0, f"expression took {elapsed:.1f}s; timeout is not bounding wall-clock"

    row0 = df["results"][0].to_list()
    assert row0[0]["is_error"] is True
    assert "TimeoutError" in row0[0]["_error"]
    assert row0[0]["tool_name"] == "t"
    # Row alignment survives a timeout: the fast call in the same row and
    # the other row are unaffected.
    assert row0[1]["is_error"] is False
    assert row0[1]["content"] == "ok"
    row1 = df["results"][1].to_list()
    assert row1[0]["is_error"] is False


def test_execute_tool_calls_python_executor_timeouts_do_not_compound():
    """N hung calls cost ~timeout_s total, not N x timeout_s."""
    def executor(tool_name, arguments):
        time.sleep(5)
        return "never used"

    df_input = pl.DataFrame({
        "calls": [json.dumps([
            {"tool_name": "t", "arguments": "{}"},
            {"tool_name": "t", "arguments": "{}"},
            {"tool_name": "t", "arguments": "{}"},
        ])],
    })

    start = time.monotonic()
    df = df_input.with_columns(
        results=execute_tool_calls(pl.col("calls"), executor=executor, timeout_s=1)
    )
    elapsed = time.monotonic() - start

    # Old code: up to 3 x 1s of sequential result() waits + 5s shutdown join.
    assert elapsed < 3.0, f"expression took {elapsed:.1f}s; timeouts are compounding"
    results = df["results"][0].to_list()
    assert len(results) == 3
    assert all(r["is_error"] for r in results)
    assert all("TimeoutError" in r["_error"] for r in results)
```

Timing caveat: the abandoned `time.sleep(5)` threads finish on their own
within ~5s; `ThreadPoolExecutor` threads are joined at interpreter exit, so
the pytest process may linger a few extra seconds after the last test — that
is expected and harmless, not a hang.

**Verify**: `pytest tests/test_tool_use.py -v` → 16 tests pass, including the
2 new ones, in well under 30s total.

### Step 4: Run the full keyless suite

**Verify**:
`pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
→ exit 0, no failures that were not already failing before your change (if
unsure of the baseline, run the same command on a clean checkout of `afc78da`
first via `git stash`, compare, then `git stash pop`).

## Test plan

- New tests, both in `tests/test_tool_use.py` (see Step 3 for full text):
  1. `test_execute_tool_calls_python_executor_timeout_is_data` — the
     regression this plan fixes: hung call + `timeout_s=1` returns in < 3s
     wall-clock; the hung call's cell is `is_error=True` with a
     `TimeoutError` `_error`; the fast call in the same row and the other
     row's call succeed (row alignment preserved).
  2. `test_execute_tool_calls_python_executor_timeouts_do_not_compound` —
     3 hung calls + `timeout_s=1` still return in < 3s (old code: sequential
     waits compound, then shutdown blocks ~5s).
- Structural pattern to match: `test_execute_tool_calls_python_executor`
  (`tests/test_tool_use.py:321-349`) — plain function, in-process executor
  closure, no fixtures, no API keys.
- Verification: `pytest tests/test_tool_use.py -v` → all pass (16 total).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pytest tests/test_tool_use.py -v` exits 0 with 16 passed (14 existing + 2 new), no API keys set.
- [ ] `pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` exits 0 (or shows only failures already present at `afc78da`).
- [ ] `grep -n "future.result(timeout" polar_llama/tools.py` returns no matches.
- [ ] `grep -n "cancel_futures=True" polar_llama/tools.py` returns exactly one match (in `_execute_batch_python`).
- [ ] `grep -c "with ThreadPoolExecutor" polar_llama/tools.py` prints `0`.
- [ ] `python -c "import polar_llama; d=polar_llama.execute_tool_calls.__doc__; assert 'deadline' in d and 'abandoned' in d, 'docstring not updated'"` exits 0.
- [ ] `git status --porcelain` shows modifications only to `polar_llama/tools.py` and `tests/test_tool_use.py` (ignore the pre-existing untracked root-level `test_*.py` files and `.DS_Store`).
- [ ] `plans/README.md` status row updated (only if you own the index — see executor instructions).

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `polar_llama/tools.py` or `tests/test_tool_use.py`
  changed since `afc78da`, and `_execute_batch_python` no longer matches the
  excerpt in "Current state" (in particular: if someone already replaced the
  sequential `.result(timeout=...)` loop, this plan is stale).
- `python -c "import polar_llama"` fails even after `maturin develop` inside
  the venv — the build environment is broken; do not debug the Rust build
  under this plan.
- The minimum supported Python drops below 3.9 in `pyproject.toml`
  (`requires-python`), which would make `cancel_futures=True` unavailable.
- Either new test fails twice on the wall-clock assertion after you have
  confirmed the implementation matches Step 1 exactly (a machine so loaded
  that a 1s deadline takes > 3s needs a human decision on the bound, not a
  looser guess).
- Making the fix work appears to require touching `src/expressions.rs`,
  `src/mcp.rs`, or the `_McpHttpSession`/`_McpStdioSession` classes.

## Maintenance notes

- **Semantic decision on record**: `timeout_s` on the Python-executor path is
  now a shared deadline measured from batch submission — not a per-call
  running-time budget. If per-call-from-start semantics are ever wanted
  (e.g. 100 slow-but-legitimate calls through `concurrency=8` where queued
  calls deserve their own budget), that requires tracking per-future start
  times inside `_run_one_python_call` and a polling wait; deliberately
  deferred as out of scope here.
- **Abandoned threads**: after a timeout, hung executor threads keep running
  until they return on their own; `ThreadPoolExecutor` joins its threads at
  interpreter exit, so a *permanently* hung executor will still block process
  exit (a CPython limitation — threads cannot be killed). The expression
  itself no longer hangs, which is the contract this plan establishes.
- **Rust/MCP path**: verified to already bound each `tools/call` via the
  reqwest client-wide `.timeout(timeout)` at `src/mcp.rs:50-54`. No follow-up
  needed there.
- **Reviewer focus**: (1) every `(row_idx, call_idx)` key still gets a
  `results` entry on every code path (done, timed-out, cancelled) — a missing
  key is a `KeyError` in the reassembly loop; (2) the `finally:` around
  `pool.shutdown(wait=False, cancel_futures=True)` so an exception during the
  wait cannot leak a joining shutdown; (3) the wall-clock assertions in the
  new tests are generous enough for CI (< 3.0s vs. a >= 5s failure mode).
- **Deferred follow-up**: the timing tests use real `time.sleep`; if they ever
  flake on very slow CI runners, consider marking them with a dedicated
  pytest marker rather than loosening the bound past the 5s failure signal.
