# Plan 021: Eight small independently-verifiable fixes (escaping, py.typed, 3.9 drop, misc correctness)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. The steps in this plan are INDEPENDENT of each
> other: if one step hits a STOP condition, skip that step, report it, and
> continue with the remaining steps rather than aborting the whole bundle.
> When done, update the status row for this plan in `plans/README.md` —
> unless a reviewer dispatched you and told you they maintain the index.
>
> **Drift check (run first)**:
> ```
> git diff --stat afc78da..HEAD -- src/expressions.rs src/cache.rs Cargo.toml pyproject.toml .github/workflows/CI.yml polar_llama/__init__.py polar_llama/optimize.py polar_llama/local/prefix_cache.py polar_llama/local/server_backend.py tests/test_message_arrays.py tests/test_local_prefix_cache.py tests/test_local_server_backend.py tests/test_feedback_improvements.py tests/test_optimize.py
> ```
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition **for the affected step only**
> (steps are independent — skip and report that step, do the rest).

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/011 (SOFT — pyproject.toml edits overlap; a `plans/011-*.md` file did not exist in `plans/` when this plan was written. If such a file exists and has landed, re-read `pyproject.toml` and rebase Step 5 onto its layout. If it does not exist, proceed.)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

Eight small, independent defects accumulate paper cuts across the library: a
JSON-injection bug that silently drops rows, a token estimate that overcounts
CJK text up to 4x, a redundant re-tokenization in the local-backend hot path,
missing `py.typed` despite the `Typing :: Typed` classifier (so mypy ignores
the package's annotations in downstream projects), support claims for EOL
Python 3.9, an undocumented credential-forwarding footgun when pointing the
OpenAI client at arbitrary local URLs, a `template()` fallback that silently
returns wrong output, and a `Predict` padding bug that surfaces as a confusing
Polars length error. Each fix is small and independently verifiable; landing
them as one branch clears the backlog cheaply.

## Current state

All excerpts verified against the working tree at commit `afc78da`.

- `src/expressions.rs` — Polars expression registrations; `string_to_message` builds message JSON by hand (lines 246–261). No `#[cfg(test)]` module exists in this file yet.
- `src/cache.rs` — prompt-cache grouping; byte-based token estimate at line 156; existing `#[cfg(test)] mod tests` starts at line 313 (use its `Message { role, content, cache_control }` construction style as the exemplar).
- `polar_llama/local/prefix_cache.py` — local MLX prefix cache; `verify_token_boundary` at lines 227–248; `checked_batch_caches` at lines 410–436 re-encodes the prefix once per suffix.
- `polar_llama/local/server_backend.py` — `set_local_endpoint` at lines 82–109; only sets `OPENAI_BASE_URL`, says nothing about the API key.
- `src/model_client/mod.rs` — `get_api_key` (lines 231–239) reads `OPENAI_API_KEY` from the environment for `Provider::OpenAI`; `apply_auth` (line 150) sends it as `request.bearer_auth(api_key)` on every request.
- `src/model_client/openai.rs` — `api_endpoint` (lines 57–61) targets `$OPENAI_BASE_URL/v1/chat/completions`. Net effect: whatever is in the ambient `OPENAI_API_KEY` is sent as a Bearer token to whatever URL `set_local_endpoint` pointed at.
- `polar_llama/__init__.py` — `template()` helper at lines 1464–1507; the `except TypeError` fallback silently drops kwargs.
- `polar_llama/optimize.py` — `Predict.__call__` at lines 273–301; pads short prediction lists but never handles over-length ones.
- `pyproject.toml` — `requires-python = ">=3.9"` (line 16), `"Programming Language :: Python :: 3.9"` classifier (line 24), `Typing :: Typed` classifier already present (line 36), `[local]` extra with `python_version >= '3.10'` markers (lines 49–55). **There is NO `[tool.maturin]` section in this file today.**
- `.github/workflows/CI.yml` — `linux_tests` matrix includes `"3.9"` at line 116: `python-version: ["3.9", "3.10", "3.11", "3.12"]`.
- `Cargo.toml` — line 14: `pyo3 = { version = "0.27", features = ["extension-module", "abi3-py39"] }`.
- No `polar_llama/py.typed` file exists.

Key excerpts:

`src/expressions.rs:240-261`:
```rust
#[derive(Deserialize)]
pub struct MessageKwargs {
    message_type: String,
}

// Register the string_to_message function with Polars
#[polars_expr(output_type=String)]
fn string_to_message(inputs: &[Series], kwargs: MessageKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let message_type = kwargs.message_type;

    let out: StringChunked = ca.apply(|opt_value| {
        opt_value.map(|value| {
            // Properly escape the content as a JSON string
            let escaped_value = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string());
            Cow::Owned(format!(
                "{{\"role\": \"{message_type}\", \"content\": {escaped_value}}}"
            ))
        })
    });
    Ok(out.into_series())
}
```

`src/cache.rs:155-156`:
```rust
        // Estimate token count (rough: ~4 chars per token)
        let estimated_tokens: usize = system_messages.iter().map(|m| m.content.len() / 4).sum();
```
(`m.content.len()` is BYTES; the same value flows into `estimated_prefix_tokens` at line 187.)

`polar_llama/local/prefix_cache.py:244-248`:
```python
    if prefix == "":
        return True
    prefix_tokens = list(tokenizer.encode(prefix))
    full_tokens = list(tokenizer.encode(prefix + suffix))
    return full_tokens[: len(prefix_tokens)] == prefix_tokens
```

`polar_llama/local/prefix_cache.py:427-436` (inside `checked_batch_caches`):
```python
        tok = tokenizer if tokenizer is not None else self._tokenizer
        if tok is None:
            raise ValueError("a tokenizer is required for the token-boundary check")
        cls, entry = self.match(model, prefix_text)
        if cls is not MatchClass.EXACT or entry is None:
            return None
        for suffix in suffixes:
            if not verify_token_boundary(tok, prefix_text, suffix):
                return None
        return entry.replicate_into_batch(len(suffixes))
```

`polar_llama/local/server_backend.py:82-109` (abridged — docstring Notes section at 97–104):
```python
def set_local_endpoint(base_url: str) -> None:
    """Point the existing OpenAI provider path at a local server.
    ...
    Notes
    -----
    This mutates process-wide environment state (there is no per-call/
    thread-local scoping in the Rust client), so it is not safe to use this
    to juggle multiple *different* local endpoints concurrently from the
    same process. For that, manage ``OPENAI_BASE_URL`` yourself around each
    call instead.
    """
    if not base_url or not base_url.strip():
        raise ValueError(
            "base_url must be a non-empty URL, e.g. 'http://localhost:8080'"
        )
    os.environ["OPENAI_BASE_URL"] = base_url.strip().rstrip("/")
```

`polar_llama/__init__.py:1496-1507`:
```python
    try:
        return pl.format(format_string, *args, **kwargs)
    except TypeError:
        # Fallback for older Polars versions that might not support kwargs
        if kwargs:
            # If kwargs are provided but not supported, we can't easily map them to positional
            # without parsing the format string.
            # But we can try to warn or just let it fail if we can't fix it.
            # Alternatively, we can assume the user knows what they are doing or guide them.
            # But the recommendation says "Abstract Prompt Templating... ensure the library works consistently".
            pass
        return pl.format(format_string, *args)
```

`polar_llama/optimize.py:297-301`:
```python
        height = df.height
        prediction_df = pl.DataFrame(
            {col: values + [None] * (height - len(values)) for col, values in columns.items()}
        )
        return pl.concat([df, prediction_df], how="horizontal")
```
(When `len(values) > height`, `[None] * (negative)` is `[]`, the column is
longer than `df`, and `pl.concat(..., how="horizontal")` raises a confusing
length error.)

Conventions that apply:
- Rust: `RUSTFLAGS="-Dwarnings"` is enforced in CI; keep clippy clean. Unit tests live in `#[cfg(test)] mod tests` at the bottom of each src file — exemplar: `src/cache.rs:313-357`.
- Python tests: pytest, marker `local` = CI-safe (see `pytestmark = pytest.mark.local` in `tests/test_local_prefix_cache.py`). Fakes over mocks: `FakeTokenizer` lives in `polar_llama/local/parity.py:97` (greedy longest-match tokenizer with an `encode(text) -> list` method). `tests/test_optimize.py` injects fake `inference_fn` callables (see `uppercase_backend` at line 35, `make_echo_backend` at line 22).
- The dev venv at `.venv/` is uv-managed (its `python` has **no pip module**); it already has pytest, polars 1.35.1, and an installed `polar_llama` extension, but **maturin is not installed in it** — install it with `uv` (see command table).
- IMPORTANT environment caveat: the working tree has an already-built `polar_llama` extension importable from `.venv`. After ANY Rust change (Steps 1, 2, 5's Cargo.toml edit), you must re-run `maturin develop` before Python tests reflect it.

## Commands you will need

All commands run from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Install maturin into venv | `uv pip install --python .venv/bin/python maturin` | exit 0 |
| Build extension | `source .venv/bin/activate && maturin develop` | exit 0, "Installed polar-llama" |
| Rust format | `cargo fmt --all -- --check` | exit 0, no output |
| Rust lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0 |
| Rust unit tests | `cargo test --lib` | all pass (15 pass at `afc78da`; more after this plan) |
| Python tests (full CI-safe set) | `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | exit 0, all pass |
| Single Python test file | `.venv/bin/pytest tests/<file> -q` | all pass |
| Build wheel | `source .venv/bin/activate && maturin build` | exit 0, wheel in `target/wheels/` |
| Inspect wheel | `unzip -l target/wheels/polar_llama-*.whl \| grep py.typed` | one line ending `polar_llama/py.typed` |

Never set or require `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/etc. — no step here needs live keys.

## Scope

**In scope** (the only files you should modify or create):
- `src/expressions.rs` (step 1)
- `src/cache.rs` (step 2)
- `polar_llama/local/prefix_cache.py` (step 3)
- `polar_llama/py.typed` (step 4 — create, empty)
- `pyproject.toml` (steps 4, 5)
- `.github/workflows/CI.yml` (step 5 — matrix line only)
- `Cargo.toml` (step 5 — abi3 feature only, optional)
- `polar_llama/local/server_backend.py` (step 6)
- `polar_llama/__init__.py` (step 7 — `template()` only)
- `polar_llama/optimize.py` (step 8)
- Tests: `tests/test_message_arrays.py`, `tests/test_local_prefix_cache.py`, `tests/test_local_server_backend.py`, `tests/test_feedback_improvements.py`, `tests/test_optimize.py`
- `plans/README.md` (status row only, at the end)

**Out of scope** (do NOT touch, even though they look related):
- `src/model_client/**` — step 6 is Python-side ONLY; do not change how the Rust client reads keys or endpoints (plans/007, 008, 010 own that area).
- `tests/model_client_tests.rs` — needs live API keys; never run or edit it.
- pyo3 version — pinned `<0.29` by pyo3-polars; documented in `.cargo/audit.toml`. Do NOT bump pyo3 itself (changing the `abi3-py39` → `abi3-py310` *feature* in step 5 is allowed; changing the `version = "0.27"` is not).
- The `[local]` extra's transformers/mlx-lm pin rationale comments in `pyproject.toml` lines 51–53 — keep the comment intact when editing markers.
- Any behavior change to `combine_messages`, `inference_*`, or the message wire format beyond JSON key order (see step 1 note).
- `tests/test_parallel_inference.py` (excluded from CI runs).

## Git workflow

- Branch: `advisor/021-small-fixes-bundle` (create from `main`).
- One commit per step (8 small commits) or logical groups; message style: sentence-case imperative summary, e.g. `Escape role via serde_json in string_to_message` (matches `git log` style like "Refresh cargo-audit ignore rationale for the pyo3 CVEs").
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1 (a): Build `string_to_message` JSON with serde_json, escaping the role

In `src/expressions.rs`, replace the hand-built `format!` JSON (lines 251–258)
so BOTH role and content go through serde_json. Target shape:

```rust
    let out: StringChunked = ca.apply(|opt_value| {
        opt_value.map(|value| {
            Cow::Owned(
                serde_json::json!({"role": message_type, "content": value}).to_string(),
            )
        })
    });
```

Note on key order: default serde_json serializes map keys alphabetically, so
output becomes `{"content":...,"role":...}` instead of role-first. This is
fine — every consumer parses the JSON (`crate::utils::parse_message_json`,
and the Python tests parse with `json.loads`). Do NOT add the serde_json
`preserve_order` feature just for cosmetics. If you find any test asserting
the raw serialized string of `string_to_message` output (none exist today),
STOP and report instead of rewriting it.

Add a `#[cfg(test)] mod tests` at the bottom of `src/expressions.rs` — none
exists there yet; model the module header on `src/cache.rs:313-315`. Because
`#[polars_expr]` wraps the function, test the JSON-building logic via a small
extracted helper rather than the expression entry point: extract

```rust
fn build_message_json(role: &str, content: &str) -> String {
    serde_json::json!({"role": role, "content": content}).to_string()
}
```

call it from `string_to_message`'s closure
(`Cow::Owned(build_message_json(&message_type, value))`), and unit-test:

- role containing a double quote and backslash (e.g. `he"ll\o`) → output
  parses with `serde_json::from_str::<serde_json::Value>` and round-trips
  `v["role"] == "he\"ll\\o"`, `v["content"]` intact;
- plain role `user` → parses, fields round-trip.

Also add one Python regression test in `tests/test_message_arrays.py`
(after `test_string_to_message`, same style — build a DataFrame, apply
`string_to_message(pl.col("content"), message_type='us"er')`, then
`json.loads` each cell and assert `parsed["role"] == 'us"er'` and the content
survives). At `afc78da` this produces invalid JSON, so the test fails before
the fix and passes after.

**Verify**:
1. `cargo test --lib` → all pass, including the 2 new `expressions::tests` tests.
2. `source .venv/bin/activate && maturin develop` → exit 0 (rebuild before Python test).
3. `.venv/bin/pytest tests/test_message_arrays.py -q` → all pass.
4. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0.

### Step 2 (b): Count chars, not bytes, in the cache token estimate

In `src/cache.rs` line 156, change the estimate to characters and fix the
comment (line 155) to say chars explicitly:

```rust
        // Estimate token count (rough: ~4 chars per token; chars, not bytes,
        // so CJK/emoji content is not overcounted)
        let estimated_tokens: usize =
            system_messages.iter().map(|m| m.content.chars().count() / 4).sum();
```

Add a unit test in the existing `mod tests` (starts `src/cache.rs:313`;
follow the `Message` construction style of `test_analyze_batch_with_shared_system_prompt`
at lines 379–425): build two message arrays sharing a system prompt of 400
identical CJK characters (e.g. `"分".repeat(400)` — 1200 bytes, 400 chars),
call `analyze_batch_for_caching(&messages, CacheStrategy::SystemPrompt, 50)`,
and assert the single shared group has `estimated_prefix_tokens == 100`
(= 400 chars / 4; the old bytes-based code would give 300).

**Verify**: `cargo test --lib` → all pass, including the new CJK test.

### Step 3 (c): Encode the shared prefix once per batch in `checked_batch_caches`

In `polar_llama/local/prefix_cache.py`:

1. Add a module-private helper next to `verify_token_boundary` (line 227):

```python
def _boundary_ok(tokenizer: Any, prefix_tokens: Sequence[int], prefix: str, suffix: str) -> bool:
    """Core of :func:`verify_token_boundary` with the prefix pre-encoded."""
    full_tokens = list(tokenizer.encode(prefix + suffix))
    return full_tokens[: len(prefix_tokens)] == list(prefix_tokens)
```

2. Keep the public `verify_token_boundary` signature and docstring EXACTLY as
   they are; make its body delegate:

```python
    if prefix == "":
        return True
    prefix_tokens = list(tokenizer.encode(prefix))
    return _boundary_ok(tokenizer, prefix_tokens, prefix, suffix)
```

3. In `checked_batch_caches` (loop at lines 433–435), hoist the prefix
   encoding out of the loop:

```python
        if prefix_text != "":
            prefix_tokens = list(tok.encode(prefix_text))
            for suffix in suffixes:
                if not _boundary_ok(tok, prefix_tokens, prefix_text, suffix):
                    return None
```

   (The `prefix_text == ""` case is trivially boundary-safe, matching
   `verify_token_boundary`'s empty-prefix contract at lines 244–245.)

Add a test in `tests/test_local_prefix_cache.py` under the
"PrefixStore" section (near `_store_with`, line 133). Use a counting
tokenizer wrapping `FakeTokenizer` (imported at the top of the file already):

```python
class _CountingTokenizer(FakeTokenizer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.encode_calls = 0

    def encode(self, text):
        self.encode_calls += 1
        return super().encode(text)
```

Build a store via `_store_with("sys\n", tokenizer=_CountingTokenizer())`
(note `_store_with` passes the engine's tokenizer into `PrefixStore`), reset
`tok.encode_calls = 0` after setup, call
`store.checked_batch_caches("m", "sys\n", ["a", "b", "c", "d"])`, assert the
result is a 4-element list AND `tok.encode_calls == 5` (1 prefix encode + 4
full encodes; the pre-fix code performs 8). Also assert
`verify_token_boundary` still catches the seam merge (existing tests at lines
183–189 cover this — they must keep passing unchanged).

**Verify**: `.venv/bin/pytest tests/test_local_prefix_cache.py -q` → all pass, including the new counting test.

### Step 4 (d): Ship `py.typed`

1. Create an EMPTY file `polar_llama/py.typed` (zero bytes).
2. Build a wheel and check whether maturin picked it up automatically
   (maturin auto-includes `py.typed` for mixed Rust/Python layouts in recent
   versions — verify rather than assume):
   `source .venv/bin/activate && maturin build && unzip -l target/wheels/polar_llama-*.whl | grep py.typed`
3. Only if the grep finds nothing: add to `pyproject.toml` (new section — no
   `[tool.maturin]` section exists today; put it after `[dependency-groups]`):

```toml
[tool.maturin]
include = [{ path = "polar_llama/py.typed", format = ["sdist", "wheel"] }]
```

   then delete stale wheels (`rm target/wheels/*.whl`), rebuild, and re-grep.

The `Typing :: Typed` classifier already exists (`pyproject.toml:36`) — do
not add another.

**Verify**: `unzip -l target/wheels/polar_llama-*.whl | grep py.typed` → exactly one line, path `polar_llama/py.typed`. (If multiple wheels accumulated in `target/wheels/`, delete them and rebuild once so the glob matches a single fresh wheel.)

### Step 5 (e): Drop EOL Python 3.9

If a landed `plans/011-*.md` restructured `pyproject.toml`, re-read the file
and apply the same logical edits to the new layout.

1. `pyproject.toml:16`: `requires-python = ">=3.9"` → `requires-python = ">=3.10"`.
2. `pyproject.toml:24`: delete the line `"Programming Language :: Python :: 3.9",`.
3. `pyproject.toml:49-55` (`[project.optional-dependencies] local`): remove
   the now-redundant `; python_version >= '3.10'` markers from both the `mlx`
   and `mlx-lm` entries (result: `"mlx>=0.31",` and `"mlx-lm>=0.31",`). KEEP
   the transformers-pin comment block (lines 51–53) verbatim.
4. `.github/workflows/CI.yml:116`: `python-version: ["3.9", "3.10", "3.11", "3.12"]`
   → `python-version: ["3.10", "3.11", "3.12"]`. Touch nothing else in the workflow.
5. `Cargo.toml:14` (optional wheel-size win): change feature `"abi3-py39"` →
   `"abi3-py310"` (pyo3 0.27 supports abi3-py310; keep `version = "0.27"`
   untouched). Then run `cargo check`. If it fails with an unknown-feature
   error, revert to `"abi3-py39"` (it is forward-compatible — a py39-abi3
   wheel still runs on 3.10+), note the revert in your report, and continue.

**Verify**:
1. `grep -n "3\.9" pyproject.toml .github/workflows/CI.yml` → no matches (a `3.9` inside an unrelated version string would be a false positive — inspect any hit).
2. `cargo check` → exit 0.
3. `source .venv/bin/activate && maturin develop` → exit 0 (confirms metadata still parses and, if you bumped abi3, the build still succeeds; the dev venv is Python 3.14, comfortably ≥3.10).

### Step 6 (f): Document and parameterize key forwarding in `set_local_endpoint`

Python-side ONLY — do not touch `src/model_client/`.

In `polar_llama/local/server_backend.py`, change the signature (line 82) to:

```python
def set_local_endpoint(base_url: str, api_key: Optional[str] = None) -> None:
```

(`Optional` is already imported in this module — confirm; it is used at line
128. If not imported, add it to the existing `typing` import.)

Behavior: after the existing `OPENAI_BASE_URL` assignment (line 109), add:

```python
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
```

Docstring: add an `api_key` entry to the Parameters section (suggested text:
"Optional value to set as ``OPENAI_API_KEY`` for the local server, e.g. a
dummy like ``'local'``. When omitted, whatever ``OPENAI_API_KEY`` is already
in the environment is sent to the local endpoint."), and extend the existing
Notes paragraph (lines 97–104) with an explicit warning, e.g.:

> Because the local path reuses the OpenAI provider, the Rust client sends
> the ambient ``OPENAI_API_KEY`` as a ``Authorization: Bearer`` header to
> whatever ``base_url`` points at (see ``get_api_key`` in
> ``src/model_client/mod.rs`` and ``api_endpoint`` in
> ``src/model_client/openai.rs``). If a real OpenAI key is set in your
> environment, it will be forwarded to the local server — pass
> ``api_key="local"`` (or any dummy) to overwrite it for this process, and
> note that this, too, mutates process-wide state.

Add tests in `tests/test_local_server_backend.py` (marker `@pytest.mark.local`,
same as neighbors; the module's autouse fixture `_isolated_openai_base_url`
at line 88 only clears `OPENAI_BASE_URL`, so manage `OPENAI_API_KEY` with
`monkeypatch` inside the new tests):

- `set_local_endpoint("http://localhost:9", api_key="local")` →
  `os.environ["OPENAI_API_KEY"] == "local"` and
  `os.environ["OPENAI_BASE_URL"] == "http://localhost:9"`.
- With `monkeypatch.setenv("OPENAI_API_KEY", "preexisting-dummy")`, calling
  `set_local_endpoint("http://localhost:9")` (no `api_key`) leaves
  `OPENAI_API_KEY` unchanged.

Use only obviously-fake values like `"local"` / `"preexisting-dummy"` —
never anything resembling a real credential.

**Verify**: `.venv/bin/pytest tests/test_local_server_backend.py -q` → all pass, including the 2 new tests.

### Step 7 (g): Make `template()` raise instead of silently dropping kwargs

In `polar_llama/__init__.py`, replace the `try/except` block at lines
1496–1507 with:

```python
    try:
        return pl.format(format_string, *args, **kwargs)
    except TypeError as exc:
        if kwargs:
            raise TypeError(
                "template() was called with keyword expressions, but the "
                "installed polars version's pl.format() does not accept "
                "keyword arguments. Upgrade polars, or use positional '{}' "
                "placeholders with positional expressions."
            ) from exc
        raise
```

Rationale to preserve in a short comment: retrying `pl.format(format_string, *args)`
after dropping kwargs returns an expression that silently ignores the named
fields — wrong output is worse than an error. When no kwargs were passed, the
original call and the fallback were identical anyway, so re-raising loses
nothing. Do not touch anything else in this 1500+-line file.

Add a test in `tests/test_feedback_improvements.py` next to
`test_template_helper` (line 80). Keyless: monkeypatch `pl.format` with a
positional-only stub so the kwargs path raises deterministically on any
polars version:

```python
def test_template_kwargs_unsupported_raises(monkeypatch):
    def positional_only_format(fstring, *args):  # simulates old polars
        return pl.lit("x")

    monkeypatch.setattr(pl, "format", positional_only_format)
    with pytest.raises(TypeError, match="keyword"):
        template("Hello {name}", name=pl.col("name"))
```

(`polar_llama` resolves `pl.format` at call time on the shared `polars`
module object, so the monkeypatch is seen. Also assert the positional path
still works under the stub: `template("Hello {}", pl.col("name"))` returns a
`pl.Expr` without raising.)

**Verify**: `.venv/bin/pytest tests/test_feedback_improvements.py -q` → all pass, including the new test.

### Step 8 (h): Reject over-length inference results in `Predict.__call__`

In `polar_llama/optimize.py`, `Predict.__call__` (lines 273–301): right after
`raw` is obtained (after the `if self.inference_fn ... else ...` block ending
line 281), add a descriptive guard:

```python
        if len(raw) > df.height:
            raise ValueError(
                f"inference returned {len(raw)} responses for a DataFrame of "
                f"height {df.height}; expected at most one response per row"
            )
```

(Raising beats silent truncation: a backend returning more responses than
rows is a bug upstream, and truncating would hide which rows the answers
belong to. Short lists keep the existing None-padding behavior at lines
297–300 — do not change it.)

Add a test in `tests/test_optimize.py` inside `class TestPredict` (follow the
injected-`inference_fn` pattern of `test_handles_invalid_json_responses` at
line ~117):

```python
    def test_overlength_inference_result_raises(self):
        def overlength_backend(messages, output_model):
            return ['{"answer": "x"}'] * (len(messages) + 2)

        module = Predict("question -> answer", inference_fn=overlength_backend)
        with pytest.raises(ValueError, match="responses"):
            module(pl.DataFrame({"question": ["q1", "q2"]}))
```

**Verify**: `.venv/bin/pytest tests/test_optimize.py -q` → all pass (23 tests: 22 existing + 1 new).

### Step 9: Full-suite gate

Run the complete verification battery once, after all surviving steps:

1. `cargo fmt --all -- --check` → exit 0.
2. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0.
3. `cargo test --lib` → all pass (15 at baseline + new tests from steps 1–2).
4. `source .venv/bin/activate && maturin develop` → exit 0.
5. `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` → exit 0, all pass.
6. `git status --porcelain` → only in-scope files listed under Scope appear.

## Test plan

New tests (each named in its step; summary):

- `src/expressions.rs` `mod tests`: `build_message_json` escapes quote/backslash in role; plain role round-trips. (Step 1)
- `tests/test_message_arrays.py`: `string_to_message` with `message_type='us"er'` produces parseable JSON. (Step 1)
- `src/cache.rs` `mod tests`: 400-CJK-char system prompt → `estimated_prefix_tokens == 100`, not 300. (Step 2)
- `tests/test_local_prefix_cache.py`: counting tokenizer sees N+1 (=5) encode calls for a 4-suffix batch. (Step 3)
- `tests/test_local_server_backend.py`: `api_key=` sets `OPENAI_API_KEY`; omission leaves it untouched. (Step 6)
- `tests/test_feedback_improvements.py`: kwargs + positional-only `pl.format` stub → clear `TypeError`; positional path unaffected. (Step 7)
- `tests/test_optimize.py`: over-length fake backend → `ValueError`. (Step 8)

Structural exemplars: `src/cache.rs:313-357` (Rust test module),
`tests/test_local_prefix_cache.py:133-138` (`_store_with`),
`tests/test_optimize.py:117-123` (injected bad backend),
`tests/test_local_server_backend.py:88-93` (env isolation fixture).

No test may read or require a real API key; use dummy strings only.

## Done criteria

Machine-checkable. ALL must hold (excluding any step skipped under a
per-step STOP, which must be listed in the report):

- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `cargo test --lib` exits 0; includes the new `expressions` and CJK cache tests
- [ ] `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` exits 0; includes all new Python tests named above
- [ ] `unzip -l target/wheels/polar_llama-*.whl | grep py.typed` → one match
- [ ] `grep -n '"3\.9"' pyproject.toml .github/workflows/CI.yml` → no matches
- [ ] `grep -c 'python_version' pyproject.toml` → 0
- [ ] `grep -n 'format!' src/expressions.rs | grep -i 'role'` → no matches (hand-built message JSON is gone)
- [ ] `git status --porcelain` shows only files in the "In scope" list
- [ ] `plans/README.md` status row updated (unless the dispatcher maintains the index)

## STOP conditions

Steps are independent: a STOP hit inside one step means SKIP that step,
record it in your report, and continue with the others. Stop the whole plan
only for the last two items.

- A step's "Current state" excerpt does not match the live code (drift since `afc78da`).
- Step 1: any existing test asserts the raw (unparsed) output string of `string_to_message` — key-order change would break it; skip and report.
- Step 4: after adding the `[tool.maturin] include` entry the wheel STILL lacks `py.typed` after one rebuild — skip and report (maturin version quirk; do not fight the build system).
- Step 5: a landed plans/011 changed `pyproject.toml` in a way that makes the line-level edits ambiguous, and you cannot map them confidently onto the new layout.
- Step 5: `cargo check` fails after the abi3 bump AND still fails after reverting to `abi3-py39` — report; something else is broken.
- Step 6: `set_local_endpoint` no longer writes `OPENAI_BASE_URL` (the Rust client contract changed) — skip and report.
- Any step's verification fails twice after a reasonable fix attempt.
- A fix appears to require touching an out-of-scope file (especially `src/model_client/**`) — stop that step.
- `maturin develop` fails on the unmodified tree (environment problem, not plan drift) — stop the whole plan and report.
- Baseline suite (`cargo test --lib` / pytest command above) is already red on the unmodified tree — stop the whole plan and report.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **Step 1** changes the serialized key order of single-message JSON from `{"role":...,"content":...}` to `{"content":...,"role":...}` (serde_json alphabetical maps). All in-repo consumers parse the JSON, but any downstream user doing string matching on message cells will notice. Reviewer should confirm no raw-string comparisons exist in docs/examples.
- **Step 2** only fixes the estimator's unit (chars vs bytes); the /4 heuristic is still crude for CJK (real tokenizers often emit ~1 token per CJK char). If cache grouping thresholds ever misbehave on CJK workloads, revisit with a real tokenizer count (tiktoken-rs is already a dependency — see `src/cost.rs`).
- **Step 5** bumps `requires-python` — this is a user-visible metadata change; call it out in the next release's changelog. If the abi3 feature was bumped to `abi3-py310`, wheel tags change from `cp39-abi3` to `cp310-abi3`; PyPI users on 3.9 (already unsupported) lose install ability, which is the point.
- **Step 6** documents, but does not eliminate, the process-global env-var mechanism. A per-call credential/endpoint design belongs to plans/010 (OpenAI-compatible client dedupe) territory — deferred deliberately.
- **Step 8** raises on over-length results. If a future backend legitimately returns multiple candidates per row, that feature needs an explicit API, not silent truncation.
- Reviewer focus: step 1's Rust closure borrow (`&message_type` inside `ca.apply`) compiles because the closure borrows immutably — if clippy complains, move the `json!` construction into the helper exactly as specified; and step 3's hoisted loop must preserve the "any failure disables reuse for the whole batch" semantics (return `None`, not per-row filtering).
