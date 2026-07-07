# Plan 005: Stop silently routing non-OpenAI embedding requests to OpenAI — error instead

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/model_client/mod.rs src/utils.rs src/expressions.rs polar_llama/__init__.py tests/test_embeddings.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

`create_embedding_client` silently hands every non-OpenAI provider an OpenAI
embedding client. A user calling `embedding_async(provider=Provider.GEMINI)`
gets the Gemini default model name `"text-embedding-004"` sent to
`api.openai.com` with their `OPENAI_API_KEY`. Every row 400s, `embed_one`
merely `eprintln!`s and returns `None`, and the user receives a silent all-null
embedding column — no exception, no failed expression, just missing data that
can flow into downstream pipelines unnoticed. After this plan, requesting
embeddings from any provider other than OpenAI raises a clear, Python-visible
error naming the provider, before any HTTP request is made.

## Current state

Relevant files:

- `src/model_client/mod.rs` — provider clients + `EmbeddingClient` trait
  (line 244), `ModelClientError` enum (lines 92-123), `embed_one` (lines
  488-499), `fetch_embeddings_generic` (lines 501-512), and the buggy
  `create_embedding_client` (lines 514-521). No `#[cfg(test)]` module exists
  in this file yet.
- `src/utils.rs` — the only call site of `create_embedding_client`
  (`fetch_embeddings_with_provider`, lines 124-137).
- `src/expressions.rs` — `#[polars_expr]` function `embedding_async` (lines
  581-646) and `get_default_embedding_model` (lines 569-578) which maps
  Gemini/Bedrock to model names that today get sent to OpenAI.
- `polar_llama/__init__.py` — Python `embedding_async` wrapper whose
  docstring (lines 717-724) advertises Gemini/Bedrock embedding models as
  first-class.
- `tests/test_embeddings.py` — existing embedding pytest file; most tests are
  gated on `OPENAI_API_KEY` via `skip_if_no_openai` (line 13);
  `test_empty_dataframe` (line 151) is keyless.

### The bug, `src/model_client/mod.rs:514-521`

```rust
/// Create an embedding client for the given provider and model
pub fn create_embedding_client(provider: Provider, model: &str) -> Box<dyn EmbeddingClient + Send + Sync> {
    match provider {
        Provider::OpenAI => Box::new(openai::OpenAIEmbeddingClient::new_with_model(model)),
        // Other providers can be added here as they're implemented
        _ => Box::new(openai::OpenAIEmbeddingClient::new_with_model(model)),
    }
}
```

### The silent-null path, `src/model_client/mod.rs:488-499`

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
```

(`embed_one` itself stays as-is — it handles per-row HTTP failures. The fix is
to never construct the wrong client in the first place.)

### `ModelClientError`, `src/model_client/mod.rs:92-111`

```rust
#[derive(Debug)]
pub enum ModelClientError {
    Http(u16, String),
    Serialization(serde_json::Error),
    RequestError(reqwest::Error),
    ParseError(String),
}

impl fmt::Display for ModelClientError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelClientError::Http(code, ref message) => write!(f, "HTTP Error {code}: {message}"),
            ModelClientError::Serialization(ref err) => write!(f, "Serialization Error: {err}"),
            ModelClientError::RequestError(ref err) => write!(f, "Request Error: {err}"),
            ModelClientError::ParseError(ref err) => write!(f, "Parse Error: {err}"),
        }
    }
}
```

The `Display` impl above is the ONLY exhaustive `match` on `ModelClientError`
in the codebase (verified at planning time), so adding a variant only requires
updating `Display`.

### The call site, `src/utils.rs:124-137`

```rust
/// Fetch embeddings with default provider (OpenAI) and model
pub async fn fetch_embeddings(texts: &[String]) -> Vec<Option<Vec<f64>>> {
    fetch_embeddings_with_provider(texts, Provider::OpenAI, "text-embedding-3-small").await
}

/// Fetch embeddings with specific provider and model
pub async fn fetch_embeddings_with_provider(
    texts: &[String],
    provider: Provider,
    model: &str
) -> Vec<Option<Vec<f64>>> {
    let client = create_embedding_client(provider, model);
    model_client::fetch_embeddings_generic(&*client, texts).await
}
```

`fetch_embeddings_with_provider` is called from exactly one place:
`src/expressions.rs:620`. `fetch_embeddings` has no callers outside
`src/utils.rs` (it is `pub` API surface of the lib crate; keep it compiling).

### The expression, `src/expressions.rs:569-578` and `608-621`

```rust
/// Get default embedding model for a given provider
fn get_default_embedding_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "text-embedding-3-small",
        Provider::Anthropic => "text-embedding-3-small", // Fallback to OpenAI
        Provider::Gemini => "text-embedding-004",
        Provider::Groq => "text-embedding-3-small", // Fallback to OpenAI
        Provider::Bedrock => "amazon.titan-embed-text-v1",
    }
}
```

```rust
    // Determine provider and model
    let provider = match &kwargs.provider {
        Some(provider_str) => parse_provider(provider_str).unwrap_or(Provider::OpenAI),
        None => Provider::OpenAI,
    };

    let model = kwargs
        .model
        .unwrap_or_else(|| get_default_embedding_model(provider).to_string());

    // Fetch embeddings in parallel using spawn for true parallelization
    let api_results = run_async(async move {
        fetch_embeddings_with_provider(&texts, provider, &model).await
    });
```

Note: `embedding_async` already returns `PolarsResult<Series>` (it is a
`#[polars_expr]` fn, line 581-582), so error propagation is a normal `?`.
There is an early return for empty/null-dtype input at lines 586-594 — that
path stays untouched (an empty frame with a non-OpenAI provider will still
return an empty column without erroring; acceptable).

### The Python docstring, `polar_llama/__init__.py:717-724`

```
    provider : str or Provider, optional
        The provider to use (OpenAI, Gemini, Bedrock). Default: OpenAI
    model : str, optional
        The embedding model name to use. If not specified, uses the default
        model for the provider:
        - OpenAI: "text-embedding-3-small" (1536 dimensions)
        - Gemini: "text-embedding-004" (768 dimensions)
        - Bedrock: "amazon.titan-embed-text-v1" (1536 dimensions)
```

### Conventions to match

- Expression-level errors use `PolarsError::ComputeError(... .into())` — see
  the exemplar in the same file, `src/expressions.rs:475-477`
  (`combine_messages`):
  ```rust
  return Err(PolarsError::ComputeError(
      "combine_messages requires at least one input".into(),
  ));
  ```
- Rust unit test modules follow the style of `src/cost.rs:344+`
  (`#[cfg(test)] mod tests { use super::*; ... }`).
- `OpenAIEmbeddingClient::new_with_model` (`src/model_client/openai.rs:162`)
  reads no env vars and cannot fail — constructing it in a unit test needs no
  API key.
- `Provider::as_str()` (`src/model_client/mod.rs:65-73`) returns lowercase
  names (`"gemini"`, `"bedrock"`, ...). The Python `Provider.GEMINI` pyclass
  stringifies to `"gemini"` (`src/lib.rs:52`).

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Venv (only if `.venv` missing) | `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt maturin` | exit 0 |
| Build + install plugin | `source .venv/bin/activate && maturin develop` | exit 0, ends with `Installed polar-llama-...` |
| Rust format | `cargo fmt --all -- --check` | exit 0, no output |
| Rust lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0, no warnings |
| Rust unit tests | `cargo test --lib` | exit 0, `0 failed` (do NOT run `cargo test` bare — `tests/model_client_tests.rs` needs live API keys) |
| New pytest only | `source .venv/bin/activate && pytest tests/test_embeddings.py -k unsupported_provider -v` | 1 passed |
| CI-safe Python suite | `source .venv/bin/activate && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | exit 0 (skips are fine) |

No live API key is required by any verification in this plan. Non-key-gated
embedding tests must pass without `OPENAI_API_KEY` set.

## Scope

**In scope** (the only files you should modify):
- `src/model_client/mod.rs` — `create_embedding_client`, `ModelClientError`
  (+ its `Display`), new `#[cfg(test)]` module
- `src/utils.rs` — `fetch_embeddings` / `fetch_embeddings_with_provider` signatures
- `src/expressions.rs` — `embedding_async` body + `get_default_embedding_model` only
- `polar_llama/__init__.py` — the `embedding_async` docstring only
- `tests/test_embeddings.py` — add one keyless test

**Out of scope** (do NOT touch, even though they look related):
- Implementing real Gemini/Bedrock/Anthropic/Groq embedding clients — a
  separate direction item.
- `OpenAIEmbeddingClient` behavior in `src/model_client/openai.rs`.
- Embedding request batching (covered by plans/014).
- `embed_one` / `fetch_embeddings_generic` per-row error handling (HTTP
  failures still yield nulls; that is deliberate and unchanged here).
- The chat-completion `create_client` / `resolve_provider_and_model` /
  `parse_provider` fallback-to-OpenAI behavior in `src/expressions.rs`
  (lines 58-86, 610) — silent provider-string fallback for chat is a separate
  concern; only the embedding path changes.
- README or other docs — only the one Python docstring listed above.

## Git workflow

- Branch: `advisor/005-embeddings-fail-loudly` (branched from `main`)
- Commit style: sentence-case imperative summary, e.g.
  `Error on unsupported embedding providers instead of routing to OpenAI`
  (matches history such as "Refresh cargo-audit ignore rationale for the pyo3 CVEs")
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Make `create_embedding_client` return a Result and add an error variant

In `src/model_client/mod.rs`:

1. Add a variant to `ModelClientError` (line 92-98):
   ```rust
   #[derive(Debug)]
   pub enum ModelClientError {
       Http(u16, String),
       Serialization(serde_json::Error),
       RequestError(reqwest::Error),
       ParseError(String),
       Unsupported(String),
   }
   ```
2. Add the matching `Display` arm (in the `match` at lines 100-109):
   ```rust
   ModelClientError::Unsupported(ref msg) => write!(f, "{msg}"),
   ```
3. Replace `create_embedding_client` (lines 514-521) with:
   ```rust
   /// Create an embedding client for the given provider and model.
   ///
   /// Only OpenAI embeddings are implemented today; every other provider is
   /// rejected here so callers fail loudly instead of silently sending
   /// requests to the wrong API.
   pub fn create_embedding_client(
       provider: Provider,
       model: &str,
   ) -> Result<Box<dyn EmbeddingClient + Send + Sync>, ModelClientError> {
       match provider {
           Provider::OpenAI => Ok(Box::new(openai::OpenAIEmbeddingClient::new_with_model(model))),
           other => Err(ModelClientError::Unsupported(format!(
               "embeddings are not implemented for provider '{}'; only OpenAI is currently supported",
               other.as_str()
           ))),
       }
   }
   ```

**Verify**: `cargo check` → fails ONLY in `src/utils.rs` (the call site not yet
updated). If it reports errors in any other file, a hidden exhaustive match on
`ModelClientError` exists — see STOP conditions.

### Step 2: Propagate the Result through `src/utils.rs`

Replace lines 124-137 of `src/utils.rs` with:

```rust
/// Fetch embeddings with default provider (OpenAI) and model
pub async fn fetch_embeddings(texts: &[String]) -> Result<Vec<Option<Vec<f64>>>, ModelClientError> {
    fetch_embeddings_with_provider(texts, Provider::OpenAI, "text-embedding-3-small").await
}

/// Fetch embeddings with specific provider and model.
/// Errors if the provider has no embedding implementation.
pub async fn fetch_embeddings_with_provider(
    texts: &[String],
    provider: Provider,
    model: &str,
) -> Result<Vec<Option<Vec<f64>>>, ModelClientError> {
    let client = create_embedding_client(provider, model)?;
    Ok(model_client::fetch_embeddings_generic(&*client, texts).await)
}
```

(`ModelClientError` is already imported at `src/utils.rs:4`.)

**Verify**: `cargo check` → fails ONLY in `src/expressions.rs` (`embedding_async`
now receives a `Result`).

### Step 3: Surface the error at the Polars expression level

In `src/expressions.rs`, `embedding_async` (lines 608-621):

1. Replace the `api_results` binding (lines 618-621):
   ```rust
   // Fetch embeddings in parallel using spawn for true parallelization
   let api_results = run_async(async move {
       fetch_embeddings_with_provider(&texts, provider, &model).await
   })
   .map_err(|e| PolarsError::ComputeError(format!("embedding_async: {e}").into()))?;
   ```
   Leave the provider/model resolution (lines 609-616) and everything after
   `api_results` unchanged.
2. Replace `get_default_embedding_model` (lines 569-578) with:
   ```rust
   /// Default embedding model. Only OpenAI embeddings are implemented today;
   /// `create_embedding_client` rejects every other provider before the model
   /// name is used, so there are no non-OpenAI defaults here.
   fn get_default_embedding_model(_provider: Provider) -> &'static str {
       "text-embedding-3-small"
   }
   ```

**Verify**: `cargo check` → exit 0. Then `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0.

### Step 4: Add the Rust unit tests

At the very bottom of `src/model_client/mod.rs`, add (there is no existing
test module in this file; style modeled on `src/cost.rs:344+`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_embedding_client_openai_ok() {
        assert!(create_embedding_client(Provider::OpenAI, "text-embedding-3-small").is_ok());
    }

    #[test]
    fn create_embedding_client_rejects_non_openai_providers() {
        for provider in [
            Provider::Anthropic,
            Provider::Gemini,
            Provider::Groq,
            Provider::Bedrock,
        ] {
            let err = create_embedding_client(provider, "some-model")
                .expect_err("non-OpenAI providers must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(provider.as_str()),
                "error must name the provider, got: {msg}"
            );
            assert!(
                msg.contains("embeddings"),
                "error must mention embeddings, got: {msg}"
            );
        }
    }
}
```

**Verify**: `cargo test --lib` → exit 0; output includes the two new tests
passing (`create_embedding_client_openai_ok`,
`create_embedding_client_rejects_non_openai_providers`) and `0 failed`.

### Step 5: Update the Python docstring

In `polar_llama/__init__.py`, replace lines 717-724 (the `provider` and
`model` parameter docs quoted in "Current state") with:

```
    provider : str or Provider, optional
        The embedding provider to use. Only OpenAI is currently supported;
        passing any other provider raises a ComputeError. Default: OpenAI
    model : str, optional
        The embedding model name to use. Default: "text-embedding-3-small"
        (1536 dimensions). Other OpenAI models such as
        "text-embedding-3-large" (3072 dimensions) may be specified.
```

Do not change the function signature, body, or the Examples block (the
examples already use only OpenAI).

**Verify**: `source .venv/bin/activate && python -c "import polar_llama; assert 'Only OpenAI is currently supported' in polar_llama.embedding_async.__doc__"` → exit 0.
(Requires `maturin develop` to have been run at least once in this venv; run
it first if the import fails with a missing extension module.)

### Step 6: Add the keyless pytest

Rebuild the plugin so the Python test exercises the new Rust code:
`source .venv/bin/activate && maturin develop` → exit 0.

In `tests/test_embeddings.py`, add after `test_empty_dataframe` (which ends
around line 166) — note it must NOT have the `@skip_if_no_openai` decorator:

```python
def test_embedding_unsupported_provider_raises():
    """Non-OpenAI providers must fail loudly, not return a null column.

    Regression test: previously Provider.GEMINI silently got an OpenAI
    client, every row 400'd against api.openai.com, and the user received
    all-null embeddings. No API key or HTTP call is needed for this test.
    """
    from polar_llama import embedding_async, Provider

    df = pl.DataFrame({"text": ["Hello world"]})

    with pytest.raises(pl.exceptions.ComputeError, match="gemini"):
        df.with_columns(
            embeddings=embedding_async(pl.col("text"), provider=Provider.GEMINI)
        )
```

If `pl.exceptions.ComputeError` turns out not to be the raised type, inspect
the actual exception (`pytest.raises(Exception)` temporarily, print
`type(exc)`); use the concrete polars exception type raised, as long as it is
a Python-visible exception whose message contains `gemini`. If no exception
reaches Python at all, that is a STOP condition.

**Verify**: `source .venv/bin/activate && pytest tests/test_embeddings.py -k unsupported_provider -v` → `1 passed` (with `OPENAI_API_KEY` unset in the environment).

### Step 7: Full verification sweep

Run, in order:

1. `cargo fmt --all` then `cargo fmt --all -- --check` → exit 0
2. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0
3. `cargo test --lib` → exit 0, `0 failed`
4. `source .venv/bin/activate && maturin develop` → exit 0
5. `source .venv/bin/activate && pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` → exit 0 (key-gated tests will be skipped; that is expected)

**Verify**: all five commands succeed as stated.

## Test plan

- New Rust unit tests in `src/model_client/mod.rs` `#[cfg(test)] mod tests`
  (Step 4): OpenAI returns `Ok`; each of Anthropic/Gemini/Groq/Bedrock returns
  `Err` whose message names the provider and mentions embeddings. Structural
  pattern: `src/cost.rs:344+`.
- New pytest `test_embedding_unsupported_provider_raises` in
  `tests/test_embeddings.py` (Step 6): eager `with_columns` with
  `Provider.GEMINI` raises a polars `ComputeError` matching `"gemini"`,
  without any API key or HTTP call — this is the regression test for the
  original silent-null bug.
- Existing keyless tests must keep passing, notably
  `tests/test_embeddings.py::test_empty_dataframe` (empty input still returns
  an empty `List[Float64]` column via the early return, no error).
- Verification: commands in Step 7.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo test --lib` exits 0 and includes
      `create_embedding_client_rejects_non_openai_providers` passing
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `grep -n "OpenAIEmbeddingClient" src/model_client/mod.rs` shows exactly
      one constructor call inside `create_embedding_client` (the
      `Provider::OpenAI` arm) — no `_ =>` fallback constructing an OpenAI client
- [ ] `grep -n "text-embedding-004\|amazon.titan-embed" src/expressions.rs`
      returns no matches
- [ ] `env -u OPENAI_API_KEY .venv/bin/pytest tests/test_embeddings.py -k unsupported_provider -v`
      → 1 passed
- [ ] `.venv/bin/pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` exits 0
- [ ] `grep -n "Gemini" polar_llama/__init__.py | grep -i embedding` returns no
      matches in the `embedding_async` docstring (the docstring no longer
      advertises Gemini/Bedrock embedding defaults)
- [ ] `git status` shows no modified files outside the in-scope list
- [ ] `plans/README.md` status row updated (if it exists and no reviewer owns it)

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows in-scope files changed since `afc78da` and the
  "Current state" excerpts no longer match the live code.
- After Step 1, `cargo check` reports errors in files other than
  `src/utils.rs` — meaning an exhaustive `match` on `ModelClientError` exists
  somewhere this plan did not account for, or the new variant conflicts with
  serde/derive expectations.
- Error propagation from `embedding_async` is blocked by the `#[polars_expr]`
  macro signature (it should not be — the function already returns
  `PolarsResult<Series>` at `src/expressions.rs:582` — but if the macro
  rejects the `?` propagation, report the exact compiler error).
- In Step 6, no Python-visible exception is raised (the expression still
  returns nulls, or the process aborts/panics instead of raising).
- Any step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. implementing
  a Gemini embedding client to make some other test pass).

## Maintenance notes

- When a real non-OpenAI embedding client lands (separate direction item),
  extend the `match` in `create_embedding_client` with an `Ok` arm for that
  provider, restore a per-provider default in `get_default_embedding_model`,
  re-expand the Python docstring, and update
  `create_embedding_client_rejects_non_openai_providers` to drop that provider
  from the rejected list.
- Reviewer should scrutinize: (1) the error message text — it is now part of
  the observable API surface (the pytest matches on it); (2) that the empty
  input early-return in `embedding_async` still short-circuits before provider
  validation (documented behavior here: empty frame + Gemini does not error);
  (3) that `fetch_embeddings`/`fetch_embeddings_with_provider` signature
  changes are acceptable — they are `pub` Rust API of the crate, so this is a
  semver-visible change for any external Rust consumer (the crate is consumed
  via Python in practice).
- Deliberately deferred: the silent `parse_provider(...).unwrap_or(Provider::OpenAI)`
  fallback for typo'd provider strings at `src/expressions.rs:610` (and the
  chat-inference equivalent, `resolve_provider_and_model`, at
  `src/expressions.rs:73-87`). A typo like `provider="gemnii"`
  still silently becomes OpenAI *chat* behavior today; for embeddings a typo
  now falls back to OpenAI and succeeds rather than erroring. Tightening
  string parsing is a separate, behavior-visible change affecting all
  expressions, not just embeddings.
- Deliberately deferred: per-row HTTP failures still produce nulls +
  `eprintln!` (`embed_one`); making row-level failures configurable/loud is a
  different design question.
