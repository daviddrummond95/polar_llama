# Plan 006: Parse Gemini safety-blocked and truncated responses gracefully instead of hard-failing

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- src/model_client/gemini.rs src/model_client/mod.rs`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/002-ci-verification-gates.md (ordering only — the CI gates make the verification here enforceable in CI; every command in this plan is runnable locally today without it)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

The Gemini API legitimately returns responses where the normal text payload is missing: on `finishReason: "SAFETY"` the candidate has **no `content` field at all**, on `finishReason: "MAX_TOKENS"` the `content` object can arrive **without a `parts` array**, and a prompt blocked outright yields an **empty/absent `candidates` array** with only a `promptFeedback.blockReason`. Today the Rust Gemini client declares all of these fields as required in its serde structs, so the *entire body* fails to deserialize and the user of this Polars plugin gets an opaque serde error (e.g. `Serialization Error: missing field 'content' at line 1 column ...`) instead of being told their prompt was safety-blocked or truncated. After this plan lands, those routine outcomes produce a descriptive `ModelClientError::ParseError` that carries the `finishReason`/`blockReason`, and the parser is covered by offline unit tests.

## Current state

Relevant files:

- `src/model_client/gemini.rs` — the Gemini provider client; contains the rigid serde structs (lines 9–29) and `parse_response` (lines 125–134). **The only file you modify.**
- `src/model_client/mod.rs` — defines `ModelClientError` (lines 93–98) and the `From<serde_json::Error>` conversion (lines 119–123); `parse_response` is called from `execute_request` at line 179 only on HTTP-success bodies. Read-only reference.
- `src/model_client/openai.rs` — the tolerant pattern to imitate: `OpenAIMessage.content` is `Option<String>` (lines 26–31). Read-only reference.
- `src/cache.rs` lines 313+ — exemplar of the repo's inline `#[cfg(test)] mod tests { use super::*; ... }` unit-test style. Read-only reference.

The rigid structs as they exist today — `src/model_client/gemini.rs:9-29`:

```rust
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    role: String,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: String,
}
```

The current parser — `src/model_client/gemini.rs:125-134`:

```rust
    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        let response: GeminiResponse = serde_json::from_str(response_text)?;
        response
            .candidates
            .into_iter()
            .next()
            .and_then(|candidate| candidate.content.parts.into_iter().next())
            .map(|part| part.text)
            .ok_or_else(|| ModelClientError::ParseError("No response content".to_string()))
    }
```

Note: the `?` on `serde_json::from_str` converts through `From<serde_json::Error>` in `src/model_client/mod.rs:119-123` into `ModelClientError::Serialization(..)` (Display: `"Serialization Error: {err}"`), not `ParseError` — that is the opaque failure users currently see for SAFETY/MAX_TOKENS bodies.

The error enum you will construct — `src/model_client/mod.rs:93-98`:

```rust
pub enum ModelClientError {
    Http(u16, String),
    Serialization(serde_json::Error),
    RequestError(reqwest::Error),
    ParseError(String),
}
```

The tolerant pattern to follow — `src/model_client/openai.rs:26-31`:

```rust
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
}
```

Real Gemini response shapes the new code must handle (from the Gemini `generateContent` REST API):

```jsonc
// 1. Happy path
{"candidates":[{"content":{"parts":[{"text":"Hello!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2}}

// 2. Safety-blocked candidate — NO "content" key
{"candidates":[{"finishReason":"SAFETY","index":0,"safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH"}]}]}

// 3. Truncated — content present but no "parts"
{"candidates":[{"content":{"role":"model"},"finishReason":"MAX_TOKENS","index":0}]}

// 4. Prompt blocked — no "candidates" at all, only promptFeedback
{"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"HIGH"}]}}
```

Repo conventions that apply:

- Error handling: providers return `Result<String, ModelClientError>`; construct descriptive `ModelClientError::ParseError(String)` for "response parsed but has no usable text" cases.
- Unit tests live in an inline `#[cfg(test)] mod tests { use super::*; ... }` at the bottom of the same file — see `src/cache.rs:313` onward for the exemplar. There is currently **no** test module in any `src/model_client/*.rs` file; you are adding the first one to `gemini.rs`.
- CI compiles with `RUSTFLAGS=-Dwarnings`, so any new warning (unused field, etc.) is a build failure. Keep the existing `#[allow(dead_code)]` on `GeminiContent` (its `role` field is never read).
- Do NOT touch pyo3/pyo3-polars versions or Cargo.toml — dependency pins are documented tradeoffs.

## Commands you will need

All verified working at the planned-at commit, run from the repo root `/Users/daviddrummond/SideProjects/polar-llama`. None of them require API keys.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Rust unit tests | `cargo test --lib` | exit 0; baseline at `afc78da` is `15 passed; 0 failed` — after this plan, ≥19 passed |
| Only the new tests | `cargo test --lib gemini` | exit 0; ≥4 tests run, all pass |
| Format check | `cargo fmt --all -- --check` | exit 0, no output |
| Lint | `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` | exit 0, no warnings/errors |
| Scope check | `git status --porcelain` | only `src/model_client/gemini.rs` modified (plus `plans/README.md` if you update the index) |

Do NOT run `cargo test --test model_client_tests` — those are live-API integration tests that need real keys; they are out of scope.

## Scope

**In scope** (the only file you should modify):

- `src/model_client/gemini.rs` — struct changes, `parse_response` rewrite, and a new inline `#[cfg(test)]` module.

**Out of scope** (do NOT touch, even though they look related):

- `src/model_client/{openai,anthropic,groq,bedrock}.rs` and `src/model_client/mod.rs` — other providers and the shared error enum keep their current shape.
- Any retry/fallback logic, and any change to how errors are encoded into the output DataFrame (that is plans/007's error-encoding contract — do not pre-empt it).
- `tests/model_client_tests.rs` — live-API integration tests; new tests go inline in `gemini.rs` only.
- `Cargo.toml`, `.cargo/audit.toml`, CI workflows, and all Python code.

## Git workflow

- Branch: `advisor/006-gemini-tolerant-parsing` (branched from `main`).
- Commit style: sentence-case imperative summary, matching `git log` (e.g. `Refresh cargo-audit ignore rationale for the pyo3 CVEs`). Suggested single commit: `Parse Gemini safety-blocked and truncated responses gracefully`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Make the Gemini serde structs tolerant of missing fields

In `src/model_client/gemini.rs`, replace the four structs at lines 9–29 with:

```rust
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<GeminiPromptFeedback>,
}

#[derive(Debug, Deserialize)]
struct GeminiPromptFeedback {
    #[serde(rename = "blockReason")]
    block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
    #[serde(default)]
    role: String,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}
```

Rationale for each change (do not deviate):

- `candidates` gets `#[serde(default)]` so a prompt-blocked body with only `promptFeedback` (shape 4 above) still deserializes.
- `prompt_feedback`/`blockReason` is the trivially-available promptFeedback handling called out by the finding — it is used only to enrich the empty-candidates error message in Step 2, nothing more.
- `content: Option<GeminiContent>` mirrors the tolerant OpenAI pattern (`openai.rs:26-31`).
- `finish_reason` is captured so the error message can say *why* there is no text.
- `parts` and `role` default so a `MAX_TOKENS` content-without-parts body (shape 3) deserializes.
- `text: Option<String>` tolerates non-text parts.

**Verify**: `cargo build --lib 2>&1 | tail -3` → compiles. It is acceptable for this intermediate state to fail with type errors inside `parse_response` (fixed in Step 2); if so, proceed directly to Step 2 and verify there.

### Step 2: Rewrite `parse_response` to surface finishReason/blockReason

Replace the body of `fn parse_response` (currently `gemini.rs:125-134`) with:

```rust
    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        let response: GeminiResponse = serde_json::from_str(response_text)?;

        let Some(candidate) = response.candidates.into_iter().next() else {
            let reason = response
                .prompt_feedback
                .and_then(|feedback| feedback.block_reason)
                .unwrap_or_else(|| "unknown".to_string());
            return Err(ModelClientError::ParseError(format!(
                "Gemini returned no candidates (blockReason: {reason})"
            )));
        };

        let finish_reason = candidate
            .finish_reason
            .unwrap_or_else(|| "unknown".to_string());

        candidate
            .content
            .and_then(|content| content.parts.into_iter().find_map(|part| part.text))
            .ok_or_else(|| {
                ModelClientError::ParseError(format!(
                    "Gemini returned no text (finishReason: {finish_reason})"
                ))
            })
    }
```

Behavior notes:

- Truly malformed JSON still fails via `?` as `ModelClientError::Serialization` — unchanged, correct.
- `find_map(|part| part.text)` returns the first part that *has* text (the old code took the first part unconditionally); this also tolerates leading non-text parts.
- The exact error strings `"Gemini returned no candidates (blockReason: {reason})"` and `"Gemini returned no text (finishReason: {finish_reason})"` are load-bearing — the tests in Step 3 assert on their substrings.

**Verify**: `cargo build --lib 2>&1 | tail -3` → `Finished` line, no errors or warnings.

### Step 3: Add the inline unit-test module

Append to the bottom of `src/model_client/gemini.rs` (after the final `}` of the `impl ModelClient` block), following the style of `src/cache.rs:313`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> GeminiClient {
        GeminiClient::default()
    }

    #[test]
    fn parse_response_happy_path() {
        let body = r#"{"candidates":[{"content":{"parts":[{"text":"Hello!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2}}"#;
        assert_eq!(client().parse_response(body).unwrap(), "Hello!");
    }

    #[test]
    fn parse_response_safety_blocked_candidate_without_content() {
        let body = r#"{"candidates":[{"finishReason":"SAFETY","index":0,"safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH"}]}]}"#;
        let err = client().parse_response(body).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("finishReason: SAFETY"), "unexpected error: {msg}");
        assert!(matches!(err, ModelClientError::ParseError(_)));
    }

    #[test]
    fn parse_response_max_tokens_without_parts() {
        let body = r#"{"candidates":[{"content":{"role":"model"},"finishReason":"MAX_TOKENS","index":0}]}"#;
        let err = client().parse_response(body).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("finishReason: MAX_TOKENS"), "unexpected error: {msg}");
        assert!(matches!(err, ModelClientError::ParseError(_)));
    }

    #[test]
    fn parse_response_prompt_blocked_no_candidates() {
        let body = r#"{"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"HIGH"}]}}"#;
        let err = client().parse_response(body).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("blockReason: SAFETY"), "unexpected error: {msg}");
        assert!(matches!(err, ModelClientError::ParseError(_)));
    }

    #[test]
    fn parse_response_empty_candidates_array() {
        let body = r#"{"candidates":[]}"#;
        let err = client().parse_response(body).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("blockReason: unknown"), "unexpected error: {msg}");
    }

    #[test]
    fn parse_response_malformed_json_is_serialization_error() {
        let err = client().parse_response("not json").unwrap_err();
        assert!(matches!(err, ModelClientError::Serialization(_)));
    }
}
```

These tests are pure serde/string work: no network, no API keys, no env vars (constructing `GeminiClient` does not read `GEMINI_API_KEY`; only `get_api_key()` does, and it is never called here).

**Verify**: `cargo test --lib gemini` → 6 tests run, `6 passed; 0 failed`.

### Step 4: Full local gate

Run, in order, from the repo root:

1. `cargo fmt --all` (then `cargo fmt --all -- --check` → exit 0, no output)
2. `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` → exit 0
3. `cargo test --lib` → exit 0, `21 passed; 0 failed` (15 baseline + 6 new)
4. `git status --porcelain` → only `src/model_client/gemini.rs` modified (plus `plans/README.md` if you update the index)

**Verify**: all four commands succeed with the stated output.

## Test plan

- New tests (all in the new `#[cfg(test)] mod tests` in `src/model_client/gemini.rs`, listed in Step 3):
  1. happy path → extracts `"Hello!"`
  2. SAFETY candidate with no `content` key → `ParseError` containing `finishReason: SAFETY`
  3. MAX_TOKENS with `content` but no `parts` → `ParseError` containing `finishReason: MAX_TOKENS`
  4. no `candidates` key, `promptFeedback.blockReason` present → `ParseError` containing `blockReason: SAFETY`
  5. empty `candidates: []` → `ParseError` containing `blockReason: unknown`
  6. malformed JSON → still `ModelClientError::Serialization`
- Structural pattern: `src/cache.rs:313` onward (inline `mod tests`, `use super::*`).
- Verification: `cargo test --lib` → `21 passed; 0 failed`.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo test --lib` exits 0 with `21 passed; 0 failed` (≥4 of the new tests exercise Gemini parse failure shapes)
- [ ] `cargo test --lib gemini` exits 0 with `6 passed`
- [ ] `cargo fmt --all -- --check` exits 0
- [ ] `RUSTFLAGS="-Dwarnings" cargo clippy --all-features` exits 0
- [ ] `grep -n "content: GeminiContent" src/model_client/gemini.rs` returns no matches (the rigid field is gone)
- [ ] `grep -c "finishReason" src/model_client/gemini.rs` ≥ 1 (the rename attribute exists)
- [ ] `git status --porcelain` shows no modified files outside `src/model_client/gemini.rs` (and `plans/README.md` if applicable)
- [ ] `plans/README.md` status row updated (skip if your dispatcher said they maintain the index)

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `src/model_client/gemini.rs` changed since `afc78da`, or the structs at gemini.rs:9-29 / `parse_response` at gemini.rs:125-134 do not match the "Current state" excerpts.
- `ModelClientError` in `src/model_client/mod.rs` no longer has a `ParseError(String)` variant or the `From<serde_json::Error>` impl (lines 93–123) — the error-construction pattern in Step 2 would be wrong.
- `cargo test --lib` at the baseline (before your changes) does not report `15 passed` — the baseline has shifted; re-confirm which tests exist before trusting the `21 passed` target.
- Fixing this appears to require touching any out-of-scope file (other providers, `mod.rs`, retry logic, Python code).
- A step's verification fails twice after a reasonable fix attempt.

## Maintenance notes

- If streaming or multi-candidate support is ever added to the Gemini client, `parse_response` must be revisited — it deliberately takes only the first candidate and the first *text-bearing* part.
- Gemini "thinking" models can emit non-text or thought parts before the answer part; the `find_map` over `part.text` already skips those, but if Gemini adds a `thought: true` marker that should be *excluded* even when it carries text, `GeminiPart` will need that field.
- Reviewer should scrutinize: the exact error strings (`blockReason:` / `finishReason:` substrings) — plan 007's error-encoding contract may later standardize these; this plan intentionally does not define a machine-readable error encoding, only human-readable messages.
- Deferred follow-ups: retry/fallback on SAFETY or MAX_TOKENS (product decision, not a parser concern); equivalent tolerant-parsing audits for groq/bedrock providers (separate findings if warranted); the `usageMetadata` field is still ignored — cost tracking for Gemini would parse it, out of scope here.
