# Plan 018: Make structured-output parse fallbacks loud — no silent all-null structs

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- polar_llama/__init__.py tests/test_optional_fields.py`
> If either file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. **Exception**: this plan depends on
> plans/017-taxonomy-thinking-field.md, which touches the same file — after
> 017 lands, `polar_llama/__init__.py` WILL show a diff. That alone is fine.
> What must still match is the try/except structure of `_parse_json_to_struct`
> shown below (two nested try/except blocks, bare `except Exception`, final
> bare `json_decode()` fallback). If that structure differs, STOP.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/017-taxonomy-thinking-field.md (touches the same function — land 017 first)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

When a user passes `response_model=SomeModel` to `inference_async` /
`inference` / `inference_messages`, the JSON responses are parsed into a
Polars Struct by `_parse_json_to_struct`. Today, if the model's response does
not match the schema, the function falls back to `cast(dtype, strict=False)`,
which silently converts every schema-noncompliant value to `null` — a real
model failure becomes an empty cell with zero signal. And if even that cast
fails, a final fallback returns a bare inferred-schema Series whose dtype does
not match the `return_dtype` declared at the `map_batches` call sites,
producing a confusing dtype error far from the actual cause. After this plan:
schema-noncompliant values still parse (rows survive), but a
`PolarLlamaParseWarning` names the fields and value counts that were nulled;
truly unparseable input raises an honest error at the parse site.

## Current state

Relevant files:

- `polar_llama/__init__.py` (1563 lines) — main Python API. Contains
  `_parse_json_to_struct` (lines 195–215) and its three call sites.
- `tests/test_optional_fields.py` — keyless tests that guard the
  Optional-fields behavior of this exact function. Must stay green.
- `tests/test_structured_outputs.py` — existing structured-output tests, but
  they self-skip without live API keys; do NOT rely on them for verification.

The function as it exists today (`polar_llama/__init__.py:195-215`, verified
at commit `afc78da`):

```python
def _parse_json_to_struct(json_str_series: pl.Series, dtype: pl.DataType) -> pl.Series:
    """Parse a JSON string series into a struct series.

    The dtype parameter is critical - it ensures Polars uses the schema derived
    from the Pydantic model rather than inferring from data. This fixes issues
    with Optional fields that may be null in some rows but present in others.
    """
    # Pass the dtype to json_decode to use the Pydantic-derived schema
    # instead of inferring from data. This ensures Optional fields are
    # always included even when null in early rows.
    try:
        return json_str_series.str.json_decode(dtype=dtype)
    except Exception:
        # If parsing fails with schema, try without and cast
        # This handles edge cases where the response doesn't match schema
        try:
            parsed = json_str_series.str.json_decode()
            return parsed.cast(dtype, strict=False)
        except Exception:
            # Last resort: return inferred schema
            return json_str_series.str.json_decode()
```

The three call sites, all identical in shape (verified line numbers at
`afc78da` — they may shift a few lines after plan 017 lands):

- `polar_llama/__init__.py:436-439` (inside `inference_async`)
- `polar_llama/__init__.py:526-529` (inside deprecated `inference`)
- `polar_llama/__init__.py:642-645` (inside `inference_messages`)

```python
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype),
            return_dtype=struct_dtype
        )
```

Call sites are NOT modified by this plan — `map_batches` declares
`return_dtype=struct_dtype`, which is exactly why the last-resort bare
`json_decode()` (inferred dtype) is dangerous.

Facts about exception behavior, verified in this repo's `.venv`
(Python 3.14.6, polars 1.35.1) — Step 1 re-verifies them:

1. Malformed JSON + `json_decode(dtype=...)` → raises
   `polars.exceptions.ComputeError` ("error deserializing JSON: json parsing
   error: ...").
2. Malformed JSON + bare `json_decode()` → raises
   `polars.exceptions.ComputeError` ("error inferring JSON: ...").
3. Valid JSON with a type-mismatched field (e.g. string where Int64 expected)
   + `json_decode(dtype=...)` → raises `polars.exceptions.ComputeError`
   ("error deserializing value ... as numeric"). This is what triggers the
   fallback path.
4. Inferred decode then `.cast(dtype, strict=False)` → succeeds, silently
   nulling mismatched values AND filling fields absent from the inferred
   struct with all-null. The absent-field case matters: a legitimately
   missing Optional field is NOT a cast loss and must not trigger the
   warning. Only compare null counts on fields present in BOTH schemas.
5. JSON that decodes to a non-struct (e.g. `'[1, 2, 3]'` → `List(Int64)`)
   then `.cast(dtype, strict=False)` → raises
   `polars.exceptions.InvalidOperationError` ("cannot cast List type ...").
   Today this lands in the last-resort branch and returns a wrong-dtype
   Series; after this plan it propagates (honest error).
6. Corollary of (2): today's last-resort branch is only *reachable* via a
   cast failure like (5) — if the inferred `json_decode()` itself failed
   inside the second `try`, re-calling it in the last resort raises the same
   `ComputeError` anyway. So deleting the branch only changes behavior for
   decode-succeeds/cast-fails inputs, which currently produce a wrong-dtype
   Series. Low-risk deletion.

Safety context for the "raise on garbage" behavior: the Rust layer encodes
its own errors as *valid JSON* objects, and `_pydantic_to_json_schema` adds
`_error`/`_details`/`_raw` fields to the schema (see
`tests/test_optional_fields.py:166-174`, `test_error_fields_included`).
Provider/API failures therefore decode via the happy path; the raise path
only fires when a response string is not valid JSON at all.

Conventions in this file:

- `warnings.warn` is already used, with a function-local
  `import warnings` (see `polar_llama/__init__.py:479-485` — the
  `DeprecationWarning` in `inference`). This plan adds a module-level
  `import warnings` instead (cleaner; both patterns coexist fine — do not
  refactor the existing local imports).
- There is no `__all__` in `polar_llama/__init__.py`; a module-level class
  is importable as `from polar_llama import PolarLlamaParseWarning` with no
  extra export step. Tests already import private helpers the same way
  (`tests/test_optional_fields.py:8`).

Pre-existing baseline failure (NOT caused by this plan): in this dev venv,
`tests/test_local_ci_smoke.py` has 3 failures because `mlx` is installed
locally (those tests assert mlx is absent, which is only true in CI). Full
keyless suite baseline at `afc78da`: `3 failed, 190 passed, 7 skipped`.

## Commands you will need

All commands run from the repo root `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Sanity: package imports | `.venv/bin/python -c "import polar_llama; print('ok')"` | prints `ok` |
| (Only if import fails) rebuild | `source .venv/bin/activate && maturin develop` | exit 0 |
| Guard tests (must stay green) | `.venv/bin/python -m pytest tests/test_optional_fields.py -q` | `7 passed` (count may grow if plan 017 added tests there) |
| New tests | `.venv/bin/python -m pytest tests/test_parse_fallbacks.py -q` | all pass |
| Full keyless suite | `.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py -q` | no failures beyond the 3 pre-existing `tests/test_local_ci_smoke.py` failures |

No Rust changes in this plan, so `cargo` commands and `maturin develop` are
not needed unless the venv is broken. Never require live API keys.

## Scope

**In scope** (the only files you should modify):

- `polar_llama/__init__.py` — ONLY: (a) add `import warnings` at module top,
  (b) add the `PolarLlamaParseWarning` class immediately above
  `_parse_json_to_struct`, (c) rewrite the body and docstring of
  `_parse_json_to_struct`. Nothing else in this 1563-line file.
- `tests/test_parse_fallbacks.py` — create (new file).
- `plans/README.md` — status row only, per executor instructions.

**Out of scope** (do NOT touch, even though they look related):

- The three `map_batches` call sites (lines ~436-439, ~526-529, ~642-645) —
  their signatures and `return_dtype` stay as-is.
- Dict-field / taxonomy `thinking`-field handling in
  `_json_schema_to_polars_dtype` or `_create_taxonomy_pydantic_model` — that
  is plans/017-taxonomy-thinking-field.md's territory.
- `tests/test_optional_fields.py` — run it, never edit it.
- `tests/test_structured_outputs.py` — needs live keys; leave alone.
- Anything under `src/` (Rust) or `polar_llama/local/`.
- The existing function-local `import warnings` at lines ~479 and ~1556 —
  leave them.

## Git workflow

- Branch: `advisor/018-parse-fallback-signal` (branch from `main` after
  plans/017 has landed).
- Commit style: sentence-case imperative summary, matching `git log`
  (e.g. "Make structured-output parse fallbacks warn instead of silently
  nulling fields").
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Re-verify the exception types raised by the installed polars

The narrowed `except` clause is only correct if `json_decode` raises
`ComputeError` on this machine's polars. Run:

```bash
.venv/bin/python - <<'EOF'
import polars as pl
dtype = pl.Struct({"a": pl.Int64, "b": pl.Utf8})
bad = pl.Series(["not json", '{"a": 1, "b": "x"}'])
mismatch = pl.Series(['{"a": "oops", "b": "x"}', '{"a": 2, "b": "y"}'])
for label, fn in [
    ("garbage+dtype", lambda: bad.str.json_decode(dtype=dtype)),
    ("garbage+inferred", lambda: bad.str.json_decode()),
    ("mismatch+dtype", lambda: mismatch.str.json_decode(dtype=dtype)),
]:
    try:
        fn()
        print(label, "-> NO ERROR")
    except Exception as e:
        print(label, "->", type(e).__module__ + "." + type(e).__name__)
EOF
```

**Verify**: all three lines end with `polars.exceptions.ComputeError`.
Expected output (polars 1.35.1):

```
garbage+dtype -> polars.exceptions.ComputeError
garbage+inferred -> polars.exceptions.ComputeError
mismatch+dtype -> polars.exceptions.ComputeError
```

If any line prints a different exception type, STOP (see STOP conditions) —
the narrowed `except` in Step 3 would be wrong.

### Step 2: Add the module-level import and the warning class

In `polar_llama/__init__.py`:

(a) Add `import warnings` to the module-level imports. The import block at
the top currently reads (lines 1-7):

```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Type, Dict, Any, List
import json

import polars as pl
```

Add `import warnings` on its own line next to `import json`.

(b) Immediately above `def _parse_json_to_struct(` (currently line 195,
just after the end of `_json_schema_to_polars_dtype`), insert:

```python
class PolarLlamaParseWarning(UserWarning):
    """Emitted when a structured-output response did not match the expected
    schema and schema-noncompliant values were nulled while parsing."""
```

**Verify**:
`.venv/bin/python -c "from polar_llama import PolarLlamaParseWarning; print(issubclass(PolarLlamaParseWarning, UserWarning))"`
→ prints `True`.

### Step 3: Rewrite `_parse_json_to_struct`

Replace the entire body (and extend the docstring) of
`_parse_json_to_struct` with this shape. Three changes versus current code:
excepts narrowed to `pl.exceptions.ComputeError`, a batch-level warning on
the cast-fallback path, and the last-resort bare `json_decode()` deleted.

```python
def _parse_json_to_struct(json_str_series: pl.Series, dtype: pl.DataType) -> pl.Series:
    """Parse a JSON string series into a struct series.

    The dtype parameter is critical - it ensures Polars uses the schema derived
    from the Pydantic model rather than inferring from data. This fixes issues
    with Optional fields that may be null in some rows but present in others.

    If decoding with the schema fails, the series is decoded with schema
    inference and cast to ``dtype`` non-strictly: values that cannot be cast
    become null, and a ``PolarLlamaParseWarning`` is emitted (once per batch)
    naming each affected field and how many values it lost. If even inferred
    decoding fails (malformed JSON), or the inferred value cannot be cast to
    a struct at all, the underlying polars exception propagates -- an honest
    error at the parse site beats a wrong-dtype Series surfacing later.
    """
    # Pass the dtype to json_decode to use the Pydantic-derived schema
    # instead of inferring from data. This ensures Optional fields are
    # always included even when null in early rows.
    try:
        return json_str_series.str.json_decode(dtype=dtype)
    except pl.exceptions.ComputeError:
        # The response doesn't match the schema exactly. Decode with
        # inference and cast non-strictly so schema-compliant values
        # survive; warn about the values the cast nulled instead of
        # failing silently.
        parsed = json_str_series.str.json_decode()
        casted = parsed.cast(dtype, strict=False)
        if isinstance(parsed.dtype, pl.Struct) and isinstance(dtype, pl.Struct):
            # Only compare fields present in BOTH schemas: a field absent
            # from the inferred struct is a legitimately-missing Optional
            # field (all-null after the cast by design), not a cast loss.
            target_fields = {f.name for f in dtype.fields}
            shared = set(parsed.struct.fields) & target_fields
            lost = {}
            for name in sorted(shared):
                delta = (
                    casted.struct.field(name).null_count()
                    - parsed.struct.field(name).null_count()
                )
                if delta > 0:
                    lost[name] = delta
            if lost:
                details = ", ".join(f"{name} ({n} value(s))" for name, n in lost.items())
                warnings.warn(
                    "Structured-output response did not match the expected "
                    f"schema; values nulled during cast: {details}. Run "
                    "without response_model to inspect the raw responses.",
                    PolarLlamaParseWarning,
                    stacklevel=2,
                )
        return casted
```

Notes:

- `pl.exceptions.ComputeError` is reachable via the existing
  `import polars as pl` (verified: `pl.exceptions.ComputeError` resolves in
  this venv). Do not add a new `from polars.exceptions import ...` line.
- Do NOT wrap the inner `parsed = ... / casted = ...` in a try/except: if
  inferred decode raises `ComputeError` (malformed JSON) or the cast raises
  `InvalidOperationError` (non-struct JSON like a bare array), let it
  propagate. That is the fix, not an oversight.
- The warning fires at most once per call (= once per `map_batches` batch),
  never per row.
- Preserve the first docstring paragraph verbatim — the Optional-fields
  rationale is load-bearing and guarded by `tests/test_optional_fields.py`.

**Verify**: `.venv/bin/python -m pytest tests/test_optional_fields.py -q` →
all pass (7 at `afc78da`; possibly more after plan 017). If ANY test in this
file fails, STOP.

### Step 4: Create `tests/test_parse_fallbacks.py`

New keyless test file, modeled structurally on
`tests/test_optional_fields.py` (plain functions, module-level Pydantic
models, `pytest.main([__file__, "-v"])` guard). Suggested content:

```python
"""Tests for _parse_json_to_struct fallback behavior: warn on schema-noncompliant
fields, raise on unparseable input, stay silent on the happy path."""
import warnings

import pytest
import polars as pl
from pydantic import BaseModel
from typing import Optional

from polar_llama import (
    _pydantic_to_json_schema,
    _json_schema_to_polars_dtype,
    _parse_json_to_struct,
    PolarLlamaParseWarning,
)


class Record(BaseModel):
    name: str
    count: int
    note: Optional[str] = None


def _record_dtype() -> pl.Struct:
    return _json_schema_to_polars_dtype(_pydantic_to_json_schema(Record))


def test_happy_path_no_warning():
    """Schema-compliant JSON parses via the dtype path and emits no warning."""
    dtype = _record_dtype()
    s = pl.Series([
        '{"name": "a", "count": 1, "note": null}',
        '{"name": "b", "count": 2, "note": "hi"}',
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", PolarLlamaParseWarning)
        result = _parse_json_to_struct(s, dtype)
    assert result.struct.field("count").to_list() == [1, 2]
    assert result.struct.field("note").to_list() == [None, "hi"]


def test_type_mismatch_warns_and_nulls_field():
    """A type-mismatched field triggers the cast fallback: the row survives,
    the bad value becomes null, and a PolarLlamaParseWarning names the field."""
    dtype = _record_dtype()
    s = pl.Series([
        '{"name": "a", "count": "not-a-number", "note": null}',
        '{"name": "b", "count": 2, "note": "hi"}',
    ])
    with pytest.warns(PolarLlamaParseWarning, match="count"):
        result = _parse_json_to_struct(s, dtype)
    assert result.len() == 2
    assert result.struct.field("count").to_list() == [None, 2]
    # Compliant fields in the same rows survive intact.
    assert result.struct.field("name").to_list() == ["a", "b"]


def test_missing_optional_field_does_not_inflate_warning():
    """On the fallback path, a legitimately-missing Optional field must not be
    reported as a cast loss (only fields present in both schemas count)."""
    dtype = _record_dtype()
    # Row 1 forces the fallback (bad count) AND omits the optional "note" key.
    s = pl.Series([
        '{"name": "a", "count": "oops"}',
        '{"name": "b", "count": 2}',
    ])
    with pytest.warns(PolarLlamaParseWarning) as record:
        result = _parse_json_to_struct(s, dtype)
    messages = [str(w.message) for w in record]
    assert any("count" in m for m in messages)
    assert not any("note" in m for m in messages)
    assert result.struct.field("note").to_list() == [None, None]


def test_malformed_json_raises():
    """Garbage input no longer falls back to an inferred-schema Series -- it
    raises at the parse site."""
    dtype = _record_dtype()
    s = pl.Series(["this is not json", '{"name": "b", "count": 2}'])
    with pytest.raises(pl.exceptions.ComputeError):
        _parse_json_to_struct(s, dtype)


def test_non_struct_json_raises():
    """JSON that decodes to a non-struct (a bare array) cannot be cast to the
    target struct; the InvalidOperationError propagates instead of returning
    a wrong-dtype Series (the deleted last-resort fallback)."""
    dtype = _record_dtype()
    s = pl.Series(["[1, 2, 3]", "[4]"])
    with pytest.raises(pl.exceptions.InvalidOperationError):
        _parse_json_to_struct(s, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Adjust only if a test exposes a genuine mismatch with the Step 3
implementation (e.g. exact warning wording) — keep all five cases.

**Verify**: `.venv/bin/python -m pytest tests/test_parse_fallbacks.py -q` →
`5 passed`.

### Step 5: Run the guard tests and the full keyless suite

```bash
.venv/bin/python -m pytest tests/test_optional_fields.py tests/test_parse_fallbacks.py -q
.venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py -q
```

**Verify**: first command → all pass. Second command → no failures other
than the pre-existing 3 in `tests/test_local_ci_smoke.py` (those fail on any
machine where `mlx` is installed; they were failing before this change —
confirm with `git stash && <same pytest command> && git stash pop` if in
doubt).

### Step 6: Commit

```bash
git checkout -b advisor/018-parse-fallback-signal   # if not already on it
git add polar_llama/__init__.py tests/test_parse_fallbacks.py
git commit -m "Make structured-output parse fallbacks warn instead of silently nulling fields"
```

**Verify**: `git status --porcelain` shows no modified tracked files other
than (possibly) `plans/README.md`; `git log -1 --stat` lists exactly
`polar_llama/__init__.py` and `tests/test_parse_fallbacks.py`.

## Test plan

- New file `tests/test_parse_fallbacks.py` (Step 4), modeled after
  `tests/test_optional_fields.py`, five keyless cases:
  1. Happy path: schema-compliant JSON → parsed, no `PolarLlamaParseWarning`.
  2. Type-mismatched field → `pytest.warns(PolarLlamaParseWarning)` naming
     the field; row still present; bad value nulled; sibling fields intact.
  3. Fallback path with a missing Optional field → warning does NOT mention
     the missing field (no false positives on Optional-missing).
  4. Malformed JSON → raises `pl.exceptions.ComputeError`.
  5. JSON array (decodes to List) → raises
     `pl.exceptions.InvalidOperationError` (documents the deleted last-resort
     behavior change).
- Existing guard: `tests/test_optional_fields.py` unchanged and green.
- Verification: `.venv/bin/python -m pytest tests/test_optional_fields.py
  tests/test_parse_fallbacks.py -q` → all pass, including 5 new tests.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `.venv/bin/python -m pytest tests/test_parse_fallbacks.py -q` exits 0
      with 5 passed (includes the `pytest.warns` test and the garbage-input
      raise test).
- [ ] `.venv/bin/python -m pytest tests/test_optional_fields.py -q` exits 0.
- [ ] `sed -n '/^def _parse_json_to_struct/,/^def _create_taxonomy_pydantic_model/p' polar_llama/__init__.py | grep -c "except Exception"`
      outputs `0` (grep exits 1 when the count is 0 — that is the pass state;
      no bare `except Exception` remains in `_parse_json_to_struct`). At
      `afc78da` the same command outputs `2`.
- [ ] `grep -c "str.json_decode" polar_llama/__init__.py` outputs `2`
      (the dtype call and the inferred-fallback call; the third, last-resort
      call is gone). At `afc78da` the same command outputs `3`.
- [ ] `.venv/bin/python -c "from polar_llama import PolarLlamaParseWarning"` exits 0.
- [ ] Full keyless suite (`.venv/bin/python -m pytest tests -m "not local_gpu"
      --ignore=tests/test_parallel_inference.py -q`) shows no failures beyond
      the pre-existing 3 in `tests/test_local_ci_smoke.py`.
- [ ] `git status --porcelain` shows no modified files outside the in-scope
      list.
- [ ] `plans/README.md` status row updated (unless the dispatcher maintains
      the index).

## STOP conditions

Stop and report back (do not improvise) if:

- Step 1's REPL check prints anything other than
  `polars.exceptions.ComputeError` for all three cases — the installed
  polars raises a different type and the narrowed `except` would silently
  change behavior. Report the actual type(s).
- `tests/test_optional_fields.py` fails at any point after your edit — the
  narrowing broke the Optional-fields path this function exists to protect.
  Revert your edit, confirm the tests pass again, and report.
- The try/except structure of `_parse_json_to_struct` in the working tree
  does not match the "Current state" excerpt (beyond docstring/comment
  changes from plan 017) — plan 017 or other work restructured the function.
- plans/017-taxonomy-thinking-field.md has not been executed yet (no such
  plan file exists, or its README status row is not DONE) — this plan must
  land after it; report and wait.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching any of the three `map_batches` call
  sites or any file in the out-of-scope list.
- The full-suite run shows NEW failures outside
  `tests/test_local_ci_smoke.py` that persist when you `git stash` your
  changes (pre-existing drift) — report rather than fixing unrelated tests.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **Behavior change to scrutinize in review**: input that decodes to
  valid-but-non-struct JSON (e.g. the model returned a bare array) previously
  returned an inferred-dtype Series (which then hit a confusing dtype error
  at the `map_batches` boundary); it now raises
  `InvalidOperationError` inside `_parse_json_to_struct`. This is
  intentional — verify the error message is actionable enough in the PR.
- The null-count comparison only inspects **top-level** struct fields.
  Losses inside nested structs cast non-strictly are not itemized in the
  warning (the values are still nulled the same as before). If nested-struct
  responses become common, extend the comparison recursively.
- If anyone later changes the Rust layer to stop emitting its errors as
  valid JSON (`_error`/`_details`/`_raw` objects), the new raise path would
  start firing on provider failures, failing whole batches — the Rust error
  encoding and this function are now coupled; see
  plans/007-unify-error-encoding.md territory.
- If the polars pin moves past a major version, re-run the Step 1 REPL
  check: the `ComputeError` contract of `json_decode` is version-dependent.
- Deferred out of this plan: making the warning include row indices (cheap
  batch-level counts only, by design), and any per-row salvage of malformed
  JSON (e.g. decode row-by-row) — both add complexity the current failure
  data doesn't justify.
