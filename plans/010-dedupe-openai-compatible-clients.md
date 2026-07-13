# Plan 010: Collapse OpenAI/Groq copy-paste into a shared openai_compat module and remove the dead trait default for request bodies

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/model_client/mod.rs src/model_client/openai.rs src/model_client/groq.rs src/model_client/openai_compat.rs`
>
> These files WILL show changes, because plans 003, 006, 007, and 008 land
> before this one and touch the same files (e.g. retry logic in
> `send_request_structured`, mock-test hooks). That expected drift is NOT a
> STOP condition. Instead, re-verify the three load-bearing facts below
> against the live code before proceeding, and treat it as a STOP condition
> only if any of them no longer holds:
>
> 1. `src/model_client/openai.rs` and `src/model_client/groq.rs` each still
>    define their own field-identical completion structs
>    (`OpenAICompletion`/`OpenAIChoice`/`OpenAIMessage` and
>    `GroqCompletion`/`GroqChoice`/`GroqMessage`).
> 2. Both files still have their own `format_request_body` impls emitting the
>    same `response_format`/`json_schema` block.
> 3. The `ModelClient` trait in `src/model_client/mod.rs` still has a DEFAULT
>    `format_request_body` body containing a
>    `Provider::OpenAI | Provider::Groq` match arm, and all five clients
>    (openai, anthropic, gemini, groq, bedrock) still override it.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/003-provider-http-mock-tests.md, plans/006-gemini-tolerant-parsing.md, plans/007-unify-error-encoding.md, plans/008-retry-rate-limit.md (all touch the same files — they must land first; rebase around their changes)
- **Category**: tech-debt
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

`src/model_client/openai.rs` and `src/model_client/groq.rs` are near-total
copy-paste of each other: identical response-deserialization structs,
identical `parse_response` bodies, identical `format_request_body` bodies.
Any fix to one (e.g. the "no hardcoded temperature/max_tokens" compatibility
decision documented only in openai.rs) silently misses the other. Worse, the
`ModelClient` trait carries a DEFAULT `format_request_body` implementation
(`src/model_client/mod.rs:186-228`) that duplicates the OpenAI/Groq schema
branch AND an Anthropic forced-tool branch — but every one of the five
clients overrides it, so that default is dead code that drifts unnoticed.
After this plan lands: one shared `openai_compat` module owns the
OpenAI-compatible wire format, OpenAI and Groq delegate to it, and the trait
makes `format_request_body` a REQUIRED method so every future provider must
be explicit about its wire format instead of silently inheriting an
OpenAI-shaped body.

## Current state

All excerpts verified at commit `afc78da`. Line numbers may have shifted if
plans 003/006/007/008 landed — match by symbol name, not line number.

Relevant files:

- `src/model_client/mod.rs` — `ModelClient` trait, `Provider` enum, batch
  runners, `create_client` factory. Contains the dead default
  `format_request_body` (lines 186–228 at planning time).
- `src/model_client/openai.rs` — `OpenAIClient` (chat) plus
  `OpenAIEmbeddingClient` (embeddings — untouched by this plan). Duplicated
  structs at lines 10–31, `format_request_body` at 88–112, `parse_response`
  at 114–122.
- `src/model_client/groq.rs` — `GroqClient`. Duplicated structs at lines
  9–30, `format_request_body` at 84–105, `parse_response` at 107–115.
- `src/model_client/{anthropic,gemini,bedrock}.rs` — OUT OF SCOPE; each has
  its own `format_request_body` override (anthropic.rs:119, gemini.rs:98,
  bedrock.rs:143) that must keep working unchanged when the trait default is
  removed.

### The duplicated structs

`src/model_client/openai.rs:10-31`:

```rust
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAICompletion {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIChoice {
    index: i32,
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
}
```

`src/model_client/groq.rs:9-30` is field-for-field identical
(`GroqCompletion`/`GroqChoice`/`GroqMessage`).

### The duplicated `parse_response`

`src/model_client/openai.rs:114-122` (groq.rs:107-115 is identical except
for the struct name):

```rust
fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
    let completion: OpenAICompletion = serde_json::from_str(response_text)?;
    completion
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| ModelClientError::ParseError("No response content".to_string()))
}
```

### The duplicated `format_request_body`

`src/model_client/openai.rs:88-112`:

```rust
fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
    // Note: no hardcoded temperature/max_tokens — newer OpenAI models
    // (o-series, gpt-5 family) reject those parameters, so we rely on
    // provider defaults for maximum compatibility.
    let mut body = json!({
        "model": self.model_name(),
        "messages": self.format_messages(messages),
    });

    // Add structured output support if schema is provided
    if let Some(schema_str) = schema {
        if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": model_name.unwrap_or("response"),
                    "strict": true,
                    "schema": schema_value
                }
            });
        }
    }

    body
}
```

`src/model_client/groq.rs:84-105` is identical except it lacks the
temperature/max_tokens comment.

### The dead trait default

`src/model_client/mod.rs:186-228`:

```rust
/// Format the full request body including messages and model name
fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
    let formatted_messages = self.format_messages(messages);
    let mut body = serde_json::json!({
        "model": self.model_name(),
        "messages": formatted_messages
    });

    // Add structured output support based on provider
    if let Some(schema_str) = schema {
        if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
            match self.provider() {
                Provider::OpenAI | Provider::Groq => {
                    // ... response_format block, duplicate of the above ...
                },
                Provider::Anthropic => {
                    // ... forced-tool block, duplicate of anthropic.rs ...
                },
                _ => {
                    // For other providers, we'll validate post-response
                }
            }
        }
    }

    body
}
```

All five clients override this, so it is never executed. It has already
drifted (it lacks openai.rs's temperature/max_tokens rationale), and its
`_ => {}` arm would silently drop the schema for any future provider that
forgot to override.

### CRITICAL non-duplication to preserve

`format_messages` is NOT identical between the two clients and must stay
per-client:

- `src/model_client/openai.rs:67-86` maps roles
  `"system" | "user" | "assistant" | "tool"` (has a `"tool" => "tool"` arm).
- `src/model_client/groq.rs:64-82` maps only
  `"system" | "user" | "assistant"` — a `"tool"` role falls to the
  `_ => "user"` default.

Do NOT merge `format_messages`. The shared body builder must accept
already-formatted messages (a `Value`) so each client keeps its own role
mapping and wire bytes are unchanged.

### Other per-client behavior that stays where it is

Each client keeps: `provider()`, `api_endpoint()` (openai.rs honors
`OPENAI_BASE_URL`; groq.rs was a fixed URL at planning time, but
plans/003-provider-http-mock-tests.md adds a `GROQ_BASE_URL` override —
whichever form you find, leave it alone), `model_name()`,
`format_messages()`, its default-model constant. Auth is the trait-default
Bearer token for both (`apply_auth` in mod.rs:149-151) — do not touch it.

### Conventions

- Errors: `ModelClientError` variants + `?` on `serde_json::from_str` (see
  the `parse_response` excerpt above — `From<serde_json::Error>` exists at
  mod.rs:119-123). Match it.
- `serde_json = "1"` without the `preserve_order` feature (Cargo.toml:17),
  so JSON maps serialize in alphabetical key order; because old and new code
  both build bodies via `json!`, wire bytes are unchanged by construction.
  Assert `serde_json::Value` equality in tests (order-independent, robust).
- Existing Rust unit-test style: `#[cfg(test)] mod tests` at the bottom of
  the file — see `src/cache.rs:314` onward for the exemplar.
- Documented constraint you must not contradict: pyo3 stays pinned <0.29
  (`.cargo/audit.toml`) — irrelevant here, but do not "fix" it in passing.

## Commands you will need

Run from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Format check | `cargo fmt --all -- --check` | exit 0, no output |
| Lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0, no warnings |
| Rust unit tests | `cargo test --lib` | exit 0, all pass (0 failed) |
| Plans/003 mock tests | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` | `test result: ok.`, 0 failed, test file UNMODIFIED |
| Plans/008 retry mock tests | `cargo test --test retry_mock_tests` | `test result: ok.`, 0 failed, test file UNMODIFIED |
| Struct-deletion check | `grep -rn "GroqCompletion\|OpenAICompletion" src/` | no output, exit 1 |
| Dead-default check | `grep -n "Provider::OpenAI | Provider::Groq" src/model_client/mod.rs` | no output, exit 1 |

Do NOT run `cargo test --test model_client_tests` as a done criterion — the
integration tests in `tests/model_client_tests.rs` require live API keys.
Never set or require real API keys for anything in this plan.

## Scope

**In scope** (the only files you should modify):
- `src/model_client/openai_compat.rs` (create)
- `src/model_client/mod.rs` (add `mod openai_compat;`, delete the trait
  default body)
- `src/model_client/openai.rs`
- `src/model_client/groq.rs`
- `plans/README.md` (status row only, if the index exists and you maintain it)

**Out of scope** (do NOT touch, even though they look related):
- `src/model_client/anthropic.rs`, `gemini.rs`, `bedrock.rs` — they keep
  their own `format_request_body` overrides; no behavior change.
- The retry/backoff logic landed by plan 008 (wherever it lives after
  rebase, likely in `send_request_structured` in mod.rs) — leave it exactly
  as is; only the `format_request_body` default is deleted from mod.rs.
- The mock tests added by plans/003 — they are the safety net and must pass
  UNMODIFIED. Editing them to make this refactor pass defeats the purpose.
- `OpenAIEmbeddingClient` and everything below the
  "Embedding Client Implementation" banner in openai.rs.
- Any wire-format change whatsoever: request bodies and parse behavior must
  be byte-identical / behavior-identical before and after.
- Adding new providers; merging `format_messages`; changing `apply_auth`,
  endpoints, env-var handling, or default models.

## Git workflow

- Branch off main: `git checkout -b advisor/010-dedupe-openai-compatible-clients`
- Commit per step, sentence-case imperative summaries matching the repo's
  log style (e.g. "Add shared openai_compat module for OpenAI-style chat
  bodies", cf. `git log --oneline`: "Refresh cargo-audit ignore rationale
  for the pyo3 CVEs").
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Preflight — confirm dependencies and green baseline

1. Confirm plans 003, 006, 007, 008 are DONE in `plans/README.md` (or that
   their changes are visibly in the tree: `tests/provider_contract_tests.rs`
   from plans/003 must exist — `ls tests/provider_contract_tests.rs` succeeds).
   `<MOCK-TESTS>` below means running BOTH:
   - `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests`
   - `cargo test --test retry_mock_tests` (added by plans/008; if that test
     target does not exist, confirm plan 008's status in `plans/README.md` —
     DONE with a different test name means use that name; not DONE means STOP,
     dependency unmet)
   If the executor of plan 003 named the test file differently, take the
   actual name from `plans/003-provider-http-mock-tests.md`'s status row or
   `ls tests/*.rs`, and use it consistently wherever `<MOCK-TESTS>` appears.
2. Run the drift check from the header and re-verify the three load-bearing
   facts listed there against the live code.
3. Establish the baseline: run `cargo fmt --all -- --check`,
   `RUSTFLAGS="-Dwarnings" cargo clippy --all-features`, `cargo test --lib`,
   and `<MOCK-TESTS>`.

**Verify**: all four baseline commands exit 0 BEFORE you change anything. If
any fails, STOP — the baseline is broken and this plan cannot certify itself.

### Step 1: Create `src/model_client/openai_compat.rs`

Create the file with this content shape (adjust doc comments freely; keep
the semantics exact):

```rust
//! Shared wire format for OpenAI-compatible chat completion APIs.
//!
//! OpenAI and Groq speak the same request/response shape; this module owns
//! that shape once so the two clients cannot drift apart. Each client keeps
//! its own endpoint, default model, and `format_messages` role mapping
//! (they differ: OpenAI supports the "tool" role, Groq does not).

use serde::Deserialize;
use serde_json::{json, Value};

use super::ModelClientError;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatCompletion {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
}

/// Parse an OpenAI-style chat completion response, returning the first
/// choice's message content.
pub(crate) fn parse_chat_response(response_text: &str) -> Result<String, ModelClientError> {
    let completion: ChatCompletion = serde_json::from_str(response_text)?;
    completion
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| ModelClientError::ParseError("No response content".to_string()))
}

/// Build an OpenAI-style chat request body.
///
/// Note: no hardcoded temperature/max_tokens — newer OpenAI models
/// (o-series, gpt-5 family) reject those parameters, so we rely on
/// provider defaults for maximum compatibility.
///
/// `formatted_messages` is the output of the client's own
/// `format_messages` — role mapping differs per provider and stays there.
pub(crate) fn build_chat_request_body(
    model: &str,
    formatted_messages: Value,
    schema: Option<&str>,
    response_name: Option<&str>,
) -> Value {
    let mut body = json!({
        "model": model,
        "messages": formatted_messages,
    });

    // Add structured output support if schema is provided
    if let Some(schema_str) = schema {
        if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": response_name.unwrap_or("response"),
                    "strict": true,
                    "schema": schema_value
                }
            });
        }
    }

    body
}
```

Semantics that must be preserved exactly (they match the current openai.rs
and groq.rs code verbatim):
- An unparseable `schema` string is silently ignored (no `response_format`
  key added) — do not "improve" this into an error.
- `response_name` defaults to `"response"`.
- Empty `choices` or `content: null` → `ParseError("No response content")`.
- JSON parse failure → `ModelClientError::Serialization` via `?`.

Add a `#[cfg(test)] mod tests` at the bottom (style: `src/cache.rs:314`)
with the golden tests listed in the Test plan section.

Then register the module in `src/model_client/mod.rs` next to the existing
module declarations at the top of the file (lines 1–5 at planning time):

```rust
pub mod openai;
pub mod anthropic;
pub mod gemini;
pub mod groq;
pub mod bedrock;
mod openai_compat;
```

(`mod`, not `pub mod` — the shared shape is an internal detail of
`model_client`; nothing outside the crate module should reach it.)

**Verify**: `cargo test --lib` → exit 0, and the new `openai_compat::tests`
names appear in the output as `ok`. `RUSTFLAGS="-Dwarnings" cargo clippy
--all-features` → exit 0.

### Step 2: Delegate `OpenAIClient` to the shared module

In `src/model_client/openai.rs`:

1. Delete the `OpenAICompletion`, `OpenAIChoice`, and `OpenAIMessage`
   structs (lines 10–31 at planning time).
2. Replace the body of `format_request_body` (keep the signature) with:

```rust
fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
    super::openai_compat::build_chat_request_body(
        self.model_name(),
        self.format_messages(messages),
        schema,
        model_name,
    )
}
```

3. Replace the body of `parse_response` with:

```rust
fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
    super::openai_compat::parse_chat_response(response_text)
}
```

4. Fix imports: `Deserialize`/`Serialize` are still needed by the embedding
   structs lower in the file — leave `use serde::{Deserialize, Serialize};`
   alone. `json!` is still used by `format_messages` — leave
   `use serde_json::{json, Value};` alone. Remove nothing else.
5. Do NOT touch `format_messages`, `api_endpoint`, `provider`,
   `model_name`, the constructors, or anything in the embedding section.

**Verify**: `cargo test --lib` → exit 0. `<MOCK-TESTS>` → all pass with the
plans/003 test files unmodified (`git status` shows no changes under the
mock-test paths). `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` →
exit 0.

### Step 3: Delegate `GroqClient` to the shared module

In `src/model_client/groq.rs`:

1. Delete the `GroqCompletion`, `GroqChoice`, and `GroqMessage` structs
   (lines 9–30 at planning time).
2. Replace `format_request_body` and `parse_response` bodies exactly as in
   Step 2 (same delegation code).
3. Remove the now-unused `use serde::Deserialize;` import (line 4 at
   planning time) — clippy under `-Dwarnings` will fail the build if you
   forget. Keep `use serde_json::{json, Value};` (`format_messages` still
   uses `json!`).
4. Do NOT touch `format_messages` (its role map intentionally lacks the
   `"tool"` arm), `api_endpoint`, or anything else.

**Verify**: `cargo test --lib` → exit 0. `<MOCK-TESTS>` → all pass,
unmodified. `grep -rn "GroqCompletion\|OpenAICompletion" src/` → no output,
exit code 1. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0.

### Step 4: Make `format_request_body` a required trait method

In `src/model_client/mod.rs`, replace the entire default implementation
(lines 186–228 at planning time — the block quoted in "Current state",
starting `/// Format the full request body...` and ending with the closing
brace after `body`) with a required method declaration:

```rust
/// Format the full request body including messages and model name.
///
/// Required (no default): every provider must be explicit about its wire
/// format. OpenAI-compatible providers should delegate to
/// `openai_compat::build_chat_request_body`.
fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value;
```

Decision and rationale (chosen over keeping a default that delegates to the
shared helper): all five existing clients already override the method, so a
default has zero users today; the old default's `_ => {}` arm silently
dropped the schema for unknown providers; and a compile error for a future
sixth provider is strictly safer than silently inheriting an OpenAI-shaped
body for a non-OpenAI API. A delegating default would reintroduce exactly
the silent-drift failure mode this plan removes.

This deletion must not orphan imports: mod.rs still uses `serde_json::json!`
(in `create_error_response`) and `Value` elsewhere — leave imports alone
unless clippy flags one as unused.

**Verify**: `cargo build` → exit 0 (all five clients already provide the
method, so nothing new to implement).
`grep -n "Provider::OpenAI | Provider::Groq" src/model_client/mod.rs` → no
output, exit 1. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` →
exit 0.

### Step 5: Final sweep

Run, in order: `cargo fmt --all` (then `cargo fmt --all -- --check`),
`RUSTFLAGS="-Dwarnings" cargo clippy --all-features`, `cargo test --lib`,
`<MOCK-TESTS>`, and `git status`.

**Verify**: all commands exit 0; `git status` shows modifications ONLY to
`src/model_client/{mod.rs,openai.rs,groq.rs}`, the new
`src/model_client/openai_compat.rs`, and (if applicable) the
`plans/README.md` status row.

## Test plan

New unit tests in `src/model_client/openai_compat.rs` under
`#[cfg(test)] mod tests` (structural exemplar: `src/cache.rs:314` onward).
Assert on `serde_json::Value` equality (order-independent), not strings.
These are golden-body tests: the expected values below were captured from
the PRE-refactor `OpenAIClient::format_request_body` behavior, so they pin
wire compatibility. Cases:

1. `build_body_without_schema_is_model_and_messages_only` —
   `build_chat_request_body("gpt-4o-mini", json!([{"role":"user","content":"hi"}]), None, None)`
   equals exactly
   `json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})`
   (assert full-value equality so no stray keys like `temperature` can ever
   sneak in).
2. `build_body_with_schema_adds_response_format` — with
   `schema = Some(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#)`
   and `response_name = Some("my_schema")`, the result's
   `body["response_format"]` equals exactly
   `json!({"type": "json_schema", "json_schema": {"name": "my_schema", "strict": true, "schema": {"type":"object","properties":{"a":{"type":"string"}}}}})`.
3. `build_body_with_schema_defaults_name_to_response` — same schema,
   `response_name = None` →
   `body["response_format"]["json_schema"]["name"] == "response"`.
4. `build_body_ignores_invalid_schema` — `schema = Some("not json{")` →
   result equals the no-schema body from case 1 (no `response_format` key).
5. `parse_chat_response_returns_first_choice_content` — input
   `r#"{"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"},{"index":1,"message":{"role":"assistant","content":"second"},"finish_reason":"stop"}]}"#`
   → `Ok("hello")`.
6. `parse_chat_response_errors_on_empty_choices` — input
   `r#"{"id":"x","model":"m","choices":[]}"#` → `Err(ModelClientError::ParseError(_))`
   with message containing `"No response content"`.
7. `parse_chat_response_errors_on_null_content` — input
   `r#"{"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"stop"}]}"#`
   → `Err(ModelClientError::ParseError(_))`.
8. `parse_chat_response_errors_on_invalid_json` — input `"nope"` →
   `Err(ModelClientError::Serialization(_))`.

Existing tests: the plans/003 request-shape mock tests are the wire-level
regression net — they must pass with ZERO modifications. `cargo test --lib`
must also keep the existing `cache`/`cost` unit tests green.

Verification: `cargo test --lib` → all pass including the 8 new
`openai_compat::tests`; `<MOCK-TESTS>` → all pass.

## Done criteria

Machine-checkable. ALL must hold (from the repo root):

- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `cargo test --lib` exits 0 and lists the 8 new `openai_compat` tests as `ok`
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_BASE_URL -u ANTHROPIC_BASE_URL -u GEMINI_BASE_URL -u GROQ_BASE_URL cargo test --test provider_contract_tests` exits 0, `cargo test --test retry_mock_tests` exits 0, and `git diff --name-only` shows no file under `tests/` modified
- [ ] `grep -rn "GroqCompletion\|OpenAICompletion" src/` exits 1 (no matches)
- [ ] `grep -n "Provider::OpenAI | Provider::Groq" src/model_client/mod.rs` exits 1 (the dead default's duplicate schema branch is gone)
- [ ] `grep -n "fn format_request_body" src/model_client/mod.rs` shows exactly one match, and the line ends with `-> Value;` (required method, no default body)
- [ ] `git status` shows changes only in `src/model_client/{mod.rs,openai.rs,groq.rs,openai_compat.rs}` (plus the plans/README.md status row, if maintained by you)
- [ ] No live API keys were set or required by any command above

## STOP conditions

Stop and report back (do not improvise) if:

- `plans/003-provider-http-mock-tests.md` does not exist, or the mock tests
  it describes are not present in the tree — the dependency has not landed
  and this refactor has no wire-format safety net.
- Any of the three load-bearing facts in the drift check no longer holds —
  in particular, if the OpenAI/Groq duplication has already been collapsed
  by someone else, this plan is moot; report instead of re-doing it.
- After any step, a plans/003 mock test fails with a request-body
  difference (a changed, added, or missing JSON key) — the refactor has
  altered wire behavior, which is forbidden. Do not edit the mock test to
  make it pass.
- Step 4's `cargo build` reveals a sixth `ModelClient` implementor (or a
  test double) that relied on the trait default — that type is out of this
  plan's scope decision; report it rather than writing a new
  `format_request_body` for it.
- The fix appears to require touching `anthropic.rs`, `gemini.rs`,
  `bedrock.rs`, `expressions.rs`, or any Python file.
- A step's verification fails twice after a reasonable fix attempt.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- Reviewer focus: diff the deleted `format_request_body`/`parse_response`
  bodies in openai.rs/groq.rs against `openai_compat.rs` token by token —
  the entire risk of this plan is an accidental semantic change during the
  move (e.g. errorring on invalid schema instead of ignoring it). Also
  confirm `format_messages` was NOT merged (OpenAI keeps the `"tool"` role
  arm; Groq must not gain it).
- Any future OpenAI-compatible provider (e.g. Together, Fireworks, a local
  OpenAI-proxy) should delegate to `openai_compat` the same way rather than
  pasting a third copy; the required trait method will force that decision
  at compile time.
- If OpenAI structured-output syntax changes (e.g. `response_format` v2),
  update `build_chat_request_body` once and verify both providers' mock
  tests — Groq tracks OpenAI's format but sometimes lags; if they ever
  diverge, split the schema block back out per-client rather than adding
  provider flags to `openai_compat`.
- Deliberately deferred out of this plan: deduplicating the
  Anthropic forced-tool block (it lives only in anthropic.rs now that the
  trait default is gone — single copy, nothing to dedupe); folding
  `format_messages` role maps together (they differ on purpose); and any
  retry/auth refactoring (owned by plans 006–008).
