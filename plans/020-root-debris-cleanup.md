# Plan 020: Remove root-level scratch scripts, wire pytest to tests/ only, relocate the upstream-PR bundle

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index. Do NOT write any report/summary `.md` file into the
> repo: the covered/not-covered table required by Step 3 goes into your final
> report message, not into a file.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- fix_async.py run.py test_bedrock.py Makefile pyproject.toml .gitignore patches docs/mlx_lm_1384_fix.md docs/local_mlx_gate_decision.md tests/test_provider.py tests/test_cache.py tests/test_optional_fields.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. Exception: a `Makefile` change that
> matches plans/013-claude-md-and-makefile.md having landed is expected — see
> Step 2 for how to adapt.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/013-claude-md-and-makefile.md (soft — only the Makefile `run`/`run-release` coordination in Step 2; this plan is executable before or after 013 lands)
- **Category**: tech-debt
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

The repo root has accumulated debris that is worse than clutter. `fix_async.py`
is a one-shot regex codemod that **rewrites `src/expressions.rs` in place if
anyone runs it** — a live footgun sitting in a published PyPI project.
`test_bedrock.py` and four untracked `test_*.py` scratch scripts sit outside
`tests/`, are never collected by `pytest tests`, and today a bare `pytest`
invocation at the repo root actually **crashes with INTERNALERROR** because
`examples/test_parallel_execution.py` calls `exit(1)` at import time when API
keys are absent. Finally, `patches/` holds an upstream mlx-lm PR bundle that
looks like it might be applied at build time but is not — the real mechanism
is the runtime monkeypatch `polar_llama/local/_mlx_patches.py` — so the bundle
belongs under `docs/` with a README saying exactly that. After this plan: the
root contains no runnable scratch, pytest can only ever collect `tests/`, any
scratch assertion worth keeping lives in `tests/` as a proper (skip-gated)
test, and the PR bundle is documented archive material.

## Current state

Verified against the working tree at `afc78da` (2026-07-06). Tracked root
debris (`git ls-files | grep -E '^(fix_async|run|test_bedrock)'` returns
exactly these three):

- `fix_async.py` (63 lines) — one-shot codemod. `fix_async.py:1-7`:

  ```python
  #!/usr/bin/env python3
  """Fix async blocks in expressions.rs"""
  import re

  with open('src/expressions.rs', 'r') as f:
      content = f.read()
  ```

  and `fix_async.py:60-61` writes the regex-mangled content back:

  ```python
  with open('src/expressions.rs', 'w') as f:
      f.write(content)
  ```

  Its job (adding `.await` inside `run_async(async { fetch_... })` blocks) was
  completed long ago; running it now would corrupt `src/expressions.rs`.

- `run.py` (85 lines) — ad-hoc live-API demo: builds a 10-row DataFrame,
  demonstrates `string_to_message` / `combine_messages` /
  `inference_messages` vs `inference_async` with `Provider.GROQ` and the
  retired model `llama3-8b-8192`, prints timings (`run.py:66-77`). Everything
  it demonstrates is already documented in `README.md` (see `README.md:82`,
  `README.md:107`, `README.md:112`, `README.md:505-521` — `combine_messages`
  + `inference_messages` usage) and `examples/` already has richer demos
  (e.g. `examples/performance_comparison_with_columns.py`,
  `examples/tool_use_calorie_tracker.py` which uses `inference_messages`).
  **Decision (made at planning time after reading both): delete `run.py`; do
  not port it to `examples/`.**

- `test_bedrock.py` (80 lines) — a Bedrock integration check that is actually
  **keyless** (it constructs expressions but never executes inference).
  Assertions worth keeping: `Provider.BEDROCK` exists (`test_bedrock.py:15`),
  `str(Provider.BEDROCK) == "bedrock"` (`test_bedrock.py:26` — verified true
  in the current venv), and `inference_async(...)` accepts both
  `provider='bedrock'` (string) and `provider=Provider.BEDROCK` (enum) at
  expression-construction time (`test_bedrock.py:52-65`). The CI-run suite
  covers none of these: `tests/test_provider.py` only checks that *some*
  provider attribute exists, and the `Provider.BEDROCK` assertion in
  `tests/test_parallel_inference.py:84-90` is in a file CI ignores
  (`--ignore=tests/test_parallel_inference.py`).

Untracked scratch at the root (`git status` shows all five as `??`):

- `test_cache_messages.py` (79 lines) — live Anthropic demo:
  `inference_messages` on message arrays with a >1024-token system prompt and
  `cache=True`; prints timings/responses, **no asserts**.
- `test_cache_real_llm.py` (199 lines) — live Anthropic demo:
  `inference_async` with `cache=False`, `cache=True`, and
  `CacheConfig(strategy=CacheStrategy.AUTO, min_tokens=256, ttl="5m")`
  (lines 151-155); prints timings and a cost estimate, **no asserts**.
- `test_cache_verify.py` (194 lines) — raw `requests.post` calls straight to
  `https://api.anthropic.com/v1/messages` with/without `cache_control`
  blocks; exercises **zero polar_llama code**, no asserts. Nothing to port.
- `test_optional_llm.py` (54 lines) — live Groq structured-output demo:
  `inference_async(..., provider=Provider.GROQ, model="openai/gpt-oss-120b",
  response_model=ClaimClassification)` where `ClaimClassification` has an
  `Optional[str]` field; prints results, no asserts. The keyless
  schema/parsing side is fully covered by `tests/test_optional_fields.py`
  (which already defines the identical `ClaimClassification` model at
  `tests/test_optional_fields.py:38-43`); the live round-trip is not covered
  anywhere.
- `.DS_Store` — macOS Finder debris. Also an untracked root `__pycache__/`
  directory exists. `.gitignore` (19 lines) covers `*.pyc` but has no
  `.DS_Store` or `__pycache__/` entries.

Existing coverage the ports must not duplicate:

- `tests/test_cache.py` (241 lines) — keyless tests for `CacheStrategy`,
  `CacheConfig` defaults/kwargs, `CacheMetrics`, and the `cache=` parameter
  in the signatures of `inference_async`/`inference_messages`. It does NOT
  exercise a live cached inference.
- `tests/test_optional_fields.py` (179 lines) — keyless schema-conversion and
  JSON-parsing tests including null/missing Optional fields and the
  `_error`/`_details`/`_raw` fields (`tests/test_optional_fields.py:166-174`).

Key-guard exemplar (the pattern to follow, per the finding use `pytest.skip`):
`tests/test_parallel_inference.py:93-95`:

```python
        configured = get_configured_providers()
        if not configured:
            pytest.skip("No providers configured. Add API keys to .env file (see .env.example)")
```

Pytest configuration — `pyproject.toml:64-68` is the whole
`[tool.pytest.ini_options]` section (no `testpaths`):

```toml
[tool.pytest.ini_options]
markers = [
    "local_gpu: requires Apple GPU + mlx (skipped in CI)",
    "local: local backend logic tests (CI-safe)",
]
```

Evidence of the collection problem: `.venv/bin/python -m pytest
--collect-only -q` at the repo root currently ends in
`INTERNALERROR> SystemExit: 1` raised from
`examples/test_parallel_execution.py:31` (`exit(1)` at import when keys are
missing). `.venv/bin/python -m pytest --collect-only -q tests` collects 219
tests cleanly.

Makefile — `Makefile:35-39` (the only lines this plan touches; the file is
40 lines total at `afc78da`):

```make
run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py
```

plans/013-claude-md-and-makefile.md rewrites other parts of the Makefile and
explicitly defers `run.py` deletion to this plan; its Step 3 conditions the
`run`/`run-release` targets (and their `.PHONY` entries) on `run.py`
existing. Step 2 below handles both orderings.

The patches bundle — `patches/` contains exactly 4 files (all git-tracked;
`git ls-files patches/` confirms):
`mlx_lm_1384_gemma3n_batched_shared_kv.patch`, `PR_1384.md`,
`PR_1384_SUBMIT.md`, `PR_1384_review.md`. It is NOT applied at build time:
`grep -rn 'patches/' src/ polar_llama/ Makefile pyproject.toml` returns
nothing (verified). The actual fix mechanism is the guarded, idempotent
runtime monkeypatch `polar_llama/local/_mlx_patches.py`
(`apply_gemma3n_batched_shared_kv_patch`, working around
<https://github.com/ml-explore/mlx-lm/issues/1384>, present in
mlx-lm <= 0.31.3). References to `patches/...` that must be updated when the
directory moves:

- `docs/local_mlx_gate_decision.md:124` — "Deliverables (uncommitted):
  `patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch`"
- `docs/mlx_lm_1384_fix.md:68` —
  "`patches/mlx_lm_1384_gemma3n_batched_shared_kv.patch` (applies with `patch -p1`"
- Self-references inside the bundle itself: `patches/PR_1384_SUBMIT.md`
  lines 3, 5, 8, 39, 45, 46, 93, 95, 119 and `patches/PR_1384_review.md:5`.

Other repo facts you need:

- Python venv: `.venv/` exists with the compiled extension installed;
  `.venv/bin/python --version` → Python 3.14.x. If imports of `polar_llama`
  fail, run `make install` first.
- `Cargo.lock` may show as locally modified (` M Cargo.lock`) before you start
  — that is pre-existing and unrelated. Do not commit it and do not revert it.
- A real `.env` exists at the repo root and is gitignored. NEVER read, print,
  or copy its contents. New tests must read `os.environ` only.
- CI (`.github/workflows/CI.yml`) runs
  `python -m pytest tests -v --tb=short -m "not local_gpu" --ignore=tests/test_parallel_inference.py`
  — it passes `tests` explicitly, so adding `testpaths` cannot change CI
  behavior; it only fixes bare `pytest` invocations.
- Commit message convention (from `git log`): sentence-case imperative
  summaries, e.g. `Add collapsed-prefill and batched quantized-KV to the local backend`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Build extension (only if imports fail) | `make install` | exit 0 |
| CI-safe Python suite | `.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` | exit 0 (skips OK) |
| Bare collection check | `.venv/bin/python -m pytest --collect-only -q` | after Step 4: only `tests/...` node IDs, no INTERNALERROR |
| Tracked-debris check | `git ls-files \| grep -E '^(fix_async\.py\|run\.py\|test_bedrock\.py)$'` | after Steps 1-3: no output, exit 1 |
| Makefile dry run | `make -n test` | exit 0 |
| Patch-path check | `grep -rn "patches/" src/ polar_llama/ Makefile pyproject.toml docs/` | after Step 6: no output, exit 1 |

None of the done criteria require API keys. The three new live tests skip
when keys are absent.

## Scope

**In scope** (the only files you may modify/create/delete):
- `fix_async.py` (delete)
- `run.py` (delete)
- `test_bedrock.py` (delete, after porting)
- `test_cache_messages.py`, `test_cache_real_llm.py`, `test_cache_verify.py`,
  `test_optional_llm.py`, `.DS_Store`, root `__pycache__/` (untracked —
  delete, after porting)
- `Makefile` (remove `run`/`run-release` targets only)
- `pyproject.toml` (`[tool.pytest.ini_options]` section only)
- `.gitignore` (append two entries)
- `patches/` → `docs/upstream/mlx_lm_1384/` (git mv all 4 files)
- `docs/upstream/mlx_lm_1384/README.md` (create)
- `docs/mlx_lm_1384_fix.md`, `docs/local_mlx_gate_decision.md` (path-reference
  updates only)
- `tests/test_provider.py`, `tests/test_cache.py`,
  `tests/test_optional_fields.py` (append ported tests only)
- `plans/README.md` (your status row only, if the index exists)

**Out of scope** (do NOT touch, even though they look related):
- `polar_llama/local/_mlx_patches.py` — the runtime monkeypatch is the source
  of truth and is correct as-is.
- Existing content of `tests/` beyond the appended tests — no reformatting,
  no renames, no marker changes.
- `examples/` — the planning-time decision is to delete `run.py`, not port it
  (its content is covered by README.md:82-112 and 505-521). Also do not "fix"
  `examples/test_parallel_execution.py`'s `exit(1)` — Step 4 makes it
  uncollectable, which is the intended remedy.
- `.github/workflows/*` — CI is owned by other plans.
- `src/` — no Rust changes of any kind.
- Every other Makefile target (`013` owns the rest of the Makefile).
- `Cargo.lock`, `.env`, `.env.example`.

## Git workflow

- Branch: `advisor/020-root-debris-cleanup` (branched from `main`)
- Commit style: sentence-case imperative summary, e.g.
  `Remove root scratch scripts and relocate the mlx-lm PR bundle`
- One commit per step or one commit for the whole plan — both acceptable.
- Use `git rm` for tracked deletions and `git mv` for the patches move so
  history follows the files.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Baseline

Confirm the CI-safe suite passes before any change:

```
.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py
```

**Verify**: exit 0 (skipped tests are fine). If it fails, STOP — the failure
is pre-existing and this plan must not paper over it.

### Step 1: Delete the codemod footgun

```
git rm fix_async.py
```

**Verify**: `git ls-files | grep -c '^fix_async.py$'` → `0` (grep exits 1);
`test ! -f fix_async.py && echo GONE` → `GONE`.

### Step 2: Delete run.py and remove the Makefile run targets

1. `git rm run.py`
2. Check whether plans/013 already landed: `grep -c '.PHONY' Makefile`.
   - **If `0`** (013 not landed — the Makefile matches the 40-line state at
     `afc78da`): delete lines `Makefile:35-39` exactly (the `run:` and
     `run-release:` targets quoted in "Current state"). Touch nothing else.
   - **If `1` or more** (013 landed): delete the `run:` and `run-release:`
     targets AND remove the words `run` and `run-release` from the `.PHONY`
     line, per plans/013's own maintenance note ("when plans/020 lands, it
     must remove the `run`/`run-release` targets and their `.PHONY`
     entries"). Touch nothing else.
3. Record which branch you took in your final report.

**Verify**:
- `grep -n "run.py" Makefile` → no output (exit 1)
- `grep -n "^run" Makefile` → no output (exit 1)
- `make -n test` → exit 0 (Makefile still parses)
- `make -n install` → exit 0

### Step 3: Port uncovered scratch assertions into tests/, then delete the scratch

First, enumerate. For each of the five scratch scripts (`test_bedrock.py`,
`test_cache_messages.py`, `test_cache_real_llm.py`, `test_cache_verify.py`,
`test_optional_llm.py`), read the file and list every behavior it checks
(explicitly with `assert`, or implicitly by exercising a code path and
printing). Build a covered/not-covered table for your final report with
columns: *scratch file — behavior — covered by (tests/ file::test, or
"NOT COVERED → ported to X")*. The planning-time analysis below is what you
should expect to find; if your reading disagrees, follow your reading and
flag the difference in the report.

Planning-time coverage analysis:

| Scratch behavior | Covered today? | Action |
|---|---|---|
| `Provider.BEDROCK` exists; `str(...) == "bedrock"` (test_bedrock.py:15,26) | No (only in CI-ignored test_parallel_inference.py) | Port, keyless → `tests/test_provider.py` |
| `inference_async` accepts `provider='bedrock'` string and `Provider.BEDROCK` enum at construction (test_bedrock.py:52-65) | No | Port, keyless → `tests/test_provider.py` |
| Live `inference_async` with `cache=True` on >1024-token shared prefix returns responses (test_cache_real_llm.py) | No (test_cache.py is signature-only) | Port, live skip-gated → `tests/test_cache.py` |
| Live `inference_messages` on message arrays with `CacheConfig(min_tokens=256)` returns responses (test_cache_messages.py + test_cache_real_llm.py:151-155) | No | Port, live skip-gated → `tests/test_cache.py` |
| Raw HTTP `cache_control` probe against api.anthropic.com (test_cache_verify.py) | Exercises no polar_llama code | Nothing to port; delete |
| Live Groq structured output with `Optional[str]` field round-trip (test_optional_llm.py) | No (test_optional_fields.py is keyless parsing only) | Port, live skip-gated → `tests/test_optional_fields.py` |

**3a. Append to `tests/test_provider.py`** (keyless, ported from
`test_bedrock.py`):

```python
def test_bedrock_provider_available():
    """Ported from root test_bedrock.py: Provider.BEDROCK exists and stringifies."""
    from polar_llama import Provider

    assert hasattr(Provider, "BEDROCK")
    assert str(Provider.BEDROCK) == "bedrock"


def test_bedrock_expression_construction():
    """Ported from root test_bedrock.py: inference_async accepts bedrock as
    string and as enum at expression-construction time (no API call)."""
    from polar_llama import Provider, inference_async, string_to_message

    df = pl.DataFrame({"Questions": ["What is the capital of France?"]})
    df = df.with_columns(prompt=string_to_message("Questions", message_type="user"))

    expr_str = inference_async(
        "prompt", provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0"
    )
    expr_enum = inference_async(
        "prompt", provider=Provider.BEDROCK, model="anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert expr_str is not None
    assert expr_enum is not None
```

(`tests/test_provider.py` already has `import polars as pl` at line 1.)

**3b. Append to `tests/test_cache.py`** (live, skip-gated; ported from
`test_cache_real_llm.py` and `test_cache_messages.py`). Add `import os` next
to the existing imports at the top of the file (`tests/test_cache.py:1-3`),
then append:

```python
# ---- Live smoke tests (ported from root test_cache_real_llm.py /
# test_cache_messages.py). Skip without ANTHROPIC_API_KEY; never a CI gate. ----

# >1024 tokens (Sonnet min) and >2048 tokens (Haiku min) shared prefix.
_LONG_PREFIX = (
    "You are an expert financial analyst. Consider revenue quality, margin "
    "trends, capital efficiency, balance sheet strength, free cash flow, "
    "working capital, capital allocation, management track record, "
    "competitive moat, industry dynamics, regulatory risk, and valuation. "
) * 80


def test_live_anthropic_inference_async_with_cache():
    """cache=True end-to-end: shared long prefix, responses come back non-null."""
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; live cache smoke test skipped")

    import polars as pl
    from polar_llama import Provider, inference_async

    questions = [
        "What's a key metric for SaaS companies?",
        "How do you assess management quality?",
    ]
    df = pl.DataFrame(
        {"prompt": [_LONG_PREFIX + "\n\nQuestion: " + q + "\nAnswer in one sentence." for q in questions]}
    )
    result = df.with_columns(
        response=inference_async(
            pl.col("prompt"),
            provider=Provider.ANTHROPIC,
            model="claude-haiku-4-5",
            cache=True,
        )
    )
    assert result["response"].null_count() == 0
    assert all(len(r) > 0 for r in result["response"].to_list())


def test_live_anthropic_inference_messages_with_cache_config():
    """CacheConfig(min_tokens=256) on message arrays: responses non-null."""
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; live cache smoke test skipped")

    import polars as pl
    from polar_llama import CacheConfig, CacheStrategy, Provider, inference_messages

    questions = ["What indicates a trend reversal?", "How do rate hikes affect valuations?"]
    messages_data = [
        [
            {"role": "system", "content": _LONG_PREFIX},
            {"role": "user", "content": q + " Answer in one sentence."},
        ]
        for q in questions
    ]
    df = pl.DataFrame({"messages": messages_data})
    result = df.with_columns(
        response=inference_messages(
            pl.col("messages"),
            provider=Provider.ANTHROPIC,
            model="claude-haiku-4-5",
            cache=CacheConfig(strategy=CacheStrategy.AUTO, min_tokens=256, ttl="5m"),
        )
    )
    assert result["response"].null_count() == 0
```

Notes: `claude-haiku-4-5` is the Anthropic model the repo's existing live
tests use (`tests/test_parallel_inference.py:64`); the scratch used
`claude-sonnet-4-20250514` — the cheaper model is a deliberate substitution,
and `_LONG_PREFIX` (~3400 words) clears Haiku's 2048-token cache minimum
noted in the scratch files. Do NOT assert on timing or cache-hit speedups —
that is why the scratch scripts were never tests.

**3c. Append to `tests/test_optional_fields.py`** (live, skip-gated; ported
from `test_optional_llm.py`). The file already defines `ClaimClassification`
(lines 38-43) and imports `pytest` and `polars as pl` (lines 2-3). Append
before the final `if __name__ == "__main__":` block (lines 177-178):

```python
def test_live_groq_optional_fields_roundtrip():
    """Ported from root test_optional_llm.py: a real structured-output call
    populates Optional[str] for some rows and leaves it null for others,
    without parse errors."""
    import os

    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set; live Optional-fields test skipped")

    from polar_llama import Provider, inference_async

    df = pl.DataFrame(
        {
            "statement": [
                "The sky is blue.",
                "It's like, you know, the stuff is bad or whatever.",
            ]
        }
    )
    prompt_prefix = (
        "Analyze this statement and determine:\n"
        "1. Is it a claim (a statement that can be true or false)?\n"
        "2. Does it need rewording to be clearer?\n"
        "3. If it needs rewording, provide a clearer version in reworded_text. "
        "If not, set reworded_text to null.\n"
        "4. Explain your reasoning.\n\nStatement: "
    )
    result = df.with_columns(
        classification=inference_async(
            pl.concat_str([pl.lit(prompt_prefix), pl.col("statement")]),
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b",
            response_model=ClaimClassification,
        )
    )
    field_names = [f.name for f in result["classification"].dtype.fields]
    assert "reworded_text" in field_names
    assert "reasoning" in field_names
    # No API/parse errors: the _error field must be null on every row.
    assert result["classification"].struct.field("_error").null_count() == len(df)
```

**3d. Run the keyless suite** (the two new bedrock tests must pass; the three
live tests must report as *skipped* when no keys are set in your shell):

```
.venv/bin/python -m pytest tests/test_provider.py tests/test_cache.py tests/test_optional_fields.py -v
```

**Verify**: exit 0; output shows `test_bedrock_provider_available PASSED`,
`test_bedrock_expression_construction PASSED`, and the three
`test_live_*` tests either `SKIPPED` (no keys in env) or `PASSED` (keys
present locally). Any FAILED here is a STOP condition.

**3e. Delete the scratch files** (only after 3d is green):

```
git rm test_bedrock.py
rm test_cache_messages.py test_cache_real_llm.py test_cache_verify.py test_optional_llm.py
rm .DS_Store
rm -rf __pycache__
```

**Verify**:
`ls fix_async.py run.py test_bedrock.py test_cache_messages.py test_cache_real_llm.py test_cache_verify.py test_optional_llm.py .DS_Store 2>&1 | grep -c "No such file"` → `8`.

### Step 4: Pin pytest collection to tests/

Edit `pyproject.toml`: inside the existing `[tool.pytest.ini_options]`
section (lines 64-68 quoted in "Current state"), add one line so the section
becomes:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "local_gpu: requires Apple GPU + mlx (skipped in CI)",
    "local: local backend logic tests (CI-safe)",
]
```

**Verify**:
- `grep -n 'testpaths = \["tests"\]' pyproject.toml` → one match
- `.venv/bin/python -m pytest --collect-only -q 2>&1 | tail -3` → a
  `N tests collected` line (expected 224 = the 219 collected at planning time
  + Step 3's five new tests), NO `INTERNALERROR`, and no node IDs outside
  `tests/`:
  `.venv/bin/python -m pytest --collect-only -q 2>/dev/null | grep '::' | grep -cv '^tests/'` → `0`

### Step 5: Extend .gitignore

Append to `.gitignore` (currently 19 lines, ending with the
"Test scripts (temporary validation files)" block):

```
# macOS Finder debris
.DS_Store

# Python bytecode caches
__pycache__/
```

**Verify**: `git check-ignore .DS_Store polar_llama/__pycache__` → prints
both paths, exit 0.

### Step 6: Relocate patches/ to docs/upstream/mlx_lm_1384/

1. Move the tracked bundle:

```
mkdir -p docs/upstream
git mv patches docs/upstream/mlx_lm_1384
```

2. Create `docs/upstream/mlx_lm_1384/README.md` with exactly:

```markdown
# Upstream PR bundle: mlx-lm #1384 (Gemma 3n batched generation)

This directory is an **archive of an upstream submission**, not part of the
polar-llama build. Nothing here is applied at build or install time.

The fix that polar-llama actually uses at runtime is the guarded, idempotent
monkeypatch in `polar_llama/local/_mlx_patches.py`
(`apply_gemma3n_batched_shared_kv_patch`) — that file is the source of truth.
See `docs/mlx_lm_1384_fix.md` for the root-cause analysis.

Contents:

- `mlx_lm_1384_gemma3n_batched_shared_kv.patch` — the upstream-PR candidate
  patch against mlx-lm 0.31.3 (`patch -p1` inside an mlx-lm checkout).
- `PR_1384.md` — PR title (first line) and body for the upstream submission.
- `PR_1384_SUBMIT.md` — step-by-step submission instructions.
- `PR_1384_review.md` — internal review notes on the patch.

Upstream issue: <https://github.com/ml-explore/mlx-lm/issues/1384>.

This entire directory can be deleted once mlx-lm ships a release containing
the fix (at which point `_mlx_patches.py` becomes a guarded no-op and can be
retired separately).
```

3. Update every `patches/` path reference to the new location. Replace the
   string `patches/` with `docs/upstream/mlx_lm_1384/` in exactly these
   files (verified complete at planning time — re-check with the grep below):
   - `docs/local_mlx_gate_decision.md` (line 124)
   - `docs/mlx_lm_1384_fix.md` (line 68)
   - `docs/upstream/mlx_lm_1384/PR_1384_SUBMIT.md` (lines 3, 5, 8, 39, 45,
     46, 93, 95, 119 pre-move)
   - `docs/upstream/mlx_lm_1384/PR_1384_review.md` (line 5 pre-move)

   Do NOT edit `PR_1384.md` content (it is the upstream PR body and contains
   no `patches/` paths) and do NOT touch the `.patch` file.

**Verify**:
- `ls patches/ 2>&1` → `No such file or directory`
- `ls docs/upstream/mlx_lm_1384/` → 5 entries (4 moved files + README.md)
- `grep -rn "patches/" src/ polar_llama/ Makefile pyproject.toml docs/` → no
  output, exit 1
- `grep -c "mlx-lm/issues/1384" docs/upstream/mlx_lm_1384/README.md` → `1`

### Step 7: Full regression pass

```
.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py
```

**Verify**: exit 0 (same green as Step 0, plus the new tests: 2 passing
bedrock tests, 3 skipped-or-passing live tests). Then
`git status --porcelain` → only the in-scope paths from the Scope section
appear (plus the pre-existing ` M Cargo.lock`, which you must NOT stage).

## Test plan

- New keyless tests (must pass without any keys):
  `tests/test_provider.py::test_bedrock_provider_available`,
  `tests/test_provider.py::test_bedrock_expression_construction`.
- New live skip-gated tests (skip cleanly without keys; that skip is the
  CI-visible behavior):
  `tests/test_cache.py::test_live_anthropic_inference_async_with_cache`,
  `tests/test_cache.py::test_live_anthropic_inference_messages_with_cache_config`,
  `tests/test_optional_fields.py::test_live_groq_optional_fields_roundtrip`.
- Structural pattern to match: plain functions + early `pytest.skip` on a
  missing env var, as in `tests/test_parallel_inference.py:93-95`.
- Verification: the Step 3d and Step 7 pytest commands → all pass/skip, zero
  failures. Do NOT add the new live tests to any CI workflow.

## Done criteria

Machine-checkable. ALL must hold (none require API keys):

- [ ] `git ls-files | grep -E '^(fix_async\.py|run\.py|test_bedrock\.py)$'` → no output (exit 1)
- [ ] `ls patches/ 2>&1 | grep -c "No such file"` → `1`
- [ ] `test -f docs/upstream/mlx_lm_1384/README.md && test -f docs/upstream/mlx_lm_1384/mlx_lm_1384_gemma3n_batched_shared_kv.patch` → exit 0
- [ ] `grep -rn "patches/" src/ polar_llama/ Makefile pyproject.toml docs/` → no output (exit 1)
- [ ] `grep -n 'testpaths = \["tests"\]' pyproject.toml` → exactly one match
- [ ] `.venv/bin/python -m pytest --collect-only -q 2>/dev/null | grep '::' | grep -cv '^tests/'` → `0`, and the same command's stderr contains no `INTERNALERROR`
- [ ] `.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py` → exit 0
- [ ] `git check-ignore .DS_Store` → prints `.DS_Store`, exit 0
- [ ] `grep -En '^(run|run-release):' Makefile` → no output (exit 1); `make -n test` exits 0
- [ ] `test ! -f test_cache_messages.py && test ! -f test_cache_real_llm.py && test ! -f test_cache_verify.py && test ! -f test_optional_llm.py` → exit 0
- [ ] `git status --porcelain` shows changes only to in-scope paths (Cargo.lock's pre-existing modification excepted and unstaged)
- [ ] Your final report contains the covered/not-covered table from Step 3
- [ ] `plans/README.md` status row updated (if the index file exists)

## STOP conditions

Stop and report back (do not improvise) if:

- A ported assertion FAILS in Step 3d — especially
  `str(Provider.BEDROCK) == "bedrock"` or the expression-construction test.
  A failing port may document a real regression: report it and do NOT delete
  the original scratch file (it is the evidence).
- Your Step 3 enumeration finds a scratch behavior that is genuinely not
  covered and not in the planning-time table above, and porting it would
  require touching files outside the Scope list.
- `grep -rn 'patches/' src/ polar_llama/ Makefile pyproject.toml` returns ANY
  match — the "not applied at build time" assumption is false; do not move
  the directory.
- The `Makefile` matches neither the `afc78da` excerpt in "Current state"
  nor the post-plans/013 shape described in Step 2 (a third party rewrote it).
- Step 0's baseline suite fails, or Step 7 fails with anything other than the
  failures already present in Step 0.
- The `[tool.pytest.ini_options]` section in `pyproject.toml` no longer
  matches the quoted excerpt (another plan restructured Python tooling —
  plans/011 territory; reconcile before editing).
- Any verification fails twice after a reasonable fix attempt.

## Maintenance notes

- **plans/013 interaction**: whichever of 013/020 lands second owns removing
  the `run`/`run-release` `.PHONY` entries — Step 2 here handles both orders;
  the reviewer should confirm the Makefile ends with neither target nor stale
  `.PHONY` words.
- **Live tests**: the three new `test_live_*` tests are skip-gated and are
  NOT wired into `.github/workflows/test-llm-apis.yml` (that workflow runs
  only `tests/test_parallel_inference.py`). If the operator later wants them
  in the manual live workflow, that is a one-line workflow change —
  deliberately deferred because `.github/workflows/*` is owned by other plans.
- **Model names will rot**: `claude-haiku-4-5` and `openai/gpt-oss-120b` are
  the repo's current live-test models; when providers retire them, update the
  live tests together with `tests/test_parallel_inference.py`.
- **docs/upstream/mlx_lm_1384/ is deletable**: once mlx-lm ships the #1384
  fix, delete the whole directory and retire
  `polar_llama/local/_mlx_patches.py` (separate change; the monkeypatch is
  already a guarded no-op on fixed versions).
- **Reviewer focus**: (a) the covered/not-covered table in the executor's
  report — confirm nothing with unique assertion value was deleted;
  (b) `testpaths` added without touching the `markers` list; (c) the moved
  docs still cross-reference correctly (`grep -rn "patches/" docs/` empty);
  (d) no drive-by Makefile edits beyond the two targets.
- **Deferred**: fixing `examples/test_parallel_execution.py`'s import-time
  `exit(1)` (made harmless by `testpaths`, but still ugly if run directly);
  renaming `examples/test_*.py` so they stop looking like pytest files.
