# Plan 017: Preserve dict-typed structured-output fields (taxonomy `thinking`) as JSON strings instead of silently mangling them

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat afc78da..HEAD -- polar_llama/__init__.py tests/test_dict_fields.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. (In particular: a future
> plans/018-*.md also modifies `_field_schema_to_polars_dtype` /
> `_parse_json_to_struct`. This plan was written to land FIRST. If those
> functions have already been changed, STOP and report.)

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/002-ci-verification-gates.md (soft — pytest.skip
  hygiene for the live-API taxonomy tests; NOT blocking, this plan is fully
  executable without it)
- **Category**: bug
- **Planned at**: commit `afc78da`, 2026-07-06

## Why this matters

`tag_taxonomy` promises each field a `thinking: Dict[str, str]` — the model's
per-value reasoning — but the schema-to-Polars converter maps every JSON-schema
object with `additionalProperties` (i.e. every Pydantic `Dict[str, str]` field)
to `pl.Utf8`. The LLM then returns a JSON *object* for `thinking`, which cannot
land in a Utf8 struct slot: with the polars pinned in this repo (1.35.1
observed) `json_decode(dtype=...)` raises, the fallback `cast(strict=False)`
kicks in, and the user receives a mangled, keys-dropped, non-JSON string like
`{"yes it is upbeat","no anger present"}` (on other polars versions the value
comes back null instead). Either way it is silent data loss on the feature's
headline output, while `value`/`confidence` survive — so nobody notices until
they try to read the reasoning. After this plan, ANY Pydantic model with a
`Dict[...]` field (not just the taxonomy path) gets that field back as a
faithful JSON string that round-trips through `json.loads`.

## Current state

Relevant files:

- `polar_llama/__init__.py` — the whole Python API (1563 lines). Contains the
  dtype derivation (`_field_schema_to_polars_dtype`, lines 134-193), the JSON→
  struct parser (`_parse_json_to_struct`, lines 195-215), the taxonomy model
  builder (`_create_taxonomy_pydantic_model`, lines 222-273), and
  `tag_taxonomy` (lines 997-1131) whose docstring advertises
  `thinking: Dict[str, str]`.
- `tests/test_optional_fields.py` — the exemplar keyless test file: imports
  the underscore helpers directly and exercises schema derivation + parsing
  with hand-built JSON strings. Model the new tests on it.
- `tests/test_taxonomy_tagging.py` — live-API tests (early-`return` when
  `ANTHROPIC_API_KEY` is unset). Do NOT touch it.

### The dict→Utf8 mapping, `polar_llama/__init__.py:182-193`

```python
    elif field_type == "object":
        # Check if this is a Dict[str, str] type (has additionalProperties)
        if "additionalProperties" in field_schema and field_schema["additionalProperties"] != False:
            # This is a dictionary type, just use Utf8 for now
            # (Polars doesn't have a good way to represent arbitrary dicts in structs)
            return pl.Utf8
        else:
            # Nested objects - recursively convert
            return _json_schema_to_polars_dtype(field_schema, root_schema)
    else:
        # Default to string for unknown types
        return pl.Utf8
```

This mapping is KEPT by this plan (dict fields stay `pl.Utf8`); the fix makes
the data actually fit the dtype instead of changing the dtype.

### The parser that loses the data, `polar_llama/__init__.py:195-215`

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

It has exactly three call sites, all identical `map_batches` lambdas passing
only `(s, struct_dtype)`: `polar_llama/__init__.py:436-439` (`inference_async`),
`polar_llama/__init__.py:526-529` (`inference`), and
`polar_llama/__init__.py:642-645` (`inference_messages`). The fix below changes
only the function body, so no call site changes.

### The taxonomy model that triggers it, `polar_llama/__init__.py:255-263`

```python
            # Create the field result model with dynamic thinking keys
            field_result_model = create_model(
                f"{field_name.title()}Result",
                thinking=(Dict[str, str], Field(..., description=f"Reasoning for each possible value: {', '.join(value_names)}")),
                reflection=(str, Field(..., description="Overall reflection on your analysis of this field")),
                value=(str, Field(..., description=f"Selected value from: {', '.join(value_names)}")),
                confidence=(float, Field(..., ge=0.0, le=1.0, description="Confidence in the selected value (0.0 to 1.0)"))
            )
```

### Observed behavior at planning time (polars 1.35.1, this repo's `.venv`)

Verified by running the internal helpers directly (this is the exact
characterization the new test encodes). Given the derived dtype
`Struct({'sentiment': Struct({'thinking': String, 'reflection': String, 'value': String, 'confidence': Float64}), '_error': String, '_details': String, '_raw': String})`
and a response row
`{"sentiment": {"thinking": {"positive": "yes it is upbeat", "negative": "no anger present"}, "reflection": "clearly positive", "value": "positive", "confidence": 0.9}}`:

1. `json_decode(dtype=dtype)` raises
   `ComputeError: error deserializing JSON: error deserializing value "Object({...})" as string`.
2. The fallback `json_decode()` + `cast(dtype, strict=False)` SUCCEEDS but
   yields `thinking == '{"yes it is upbeat","no anger present"}'` — the dict's
   *keys are dropped* and the result is not valid JSON (`json.loads` raises on
   it). `value` and `confidence` survive.

So on this polars the loss is a mangled string rather than the null described
in older reports — the fix and the test below handle both by asserting a
faithful `json.loads` round-trip, not merely non-null.

### Docstrings that must be updated to match the new behavior

`polar_llama/__init__.py:239-243` (inside `_create_taxonomy_pydantic_model`'s
docstring):

```
    Each field in the output model will be a struct containing:
    - thinking: Dict[str, str] - reasoning for each possible value
    - reflection: str - overall reflection on the field analysis
    - value: str - the selected value
    - confidence: float - confidence in the selection (0.0 to 1.0)
```

`polar_llama/__init__.py:1041-1047` (inside `tag_taxonomy`'s docstring,
"Returns" section):

```
    polars.Expr
        Expression with structured tags as a Struct. Each taxonomy field becomes
        a nested struct containing:
        - thinking: Dict[str, str] - reasoning for each possible value
        - reflection: str - overall reflection on the field analysis
        - value: str - the selected value
        - confidence: float - confidence score (0.0 to 1.0)
```

### Conventions to match

- Keyless tests import underscore helpers straight from the package, e.g.
  `tests/test_optional_fields.py:8`:
  `from polar_llama import _pydantic_to_json_schema, _json_schema_to_polars_dtype, _parse_json_to_struct`
  and end with `if __name__ == "__main__": pytest.main([__file__, "-v"])`.
  (`_create_taxonomy_pydantic_model` is importable the same way — it lives in
  `polar_llama/__init__.py`.)
- The codebase writes `pl.Utf8` (not `pl.String`) — see
  `polar_llama/__init__.py:168` and `tests/test_optional_fields.py:61-65`.
  Match it.
- `json` and `Optional` are already imported at `polar_llama/__init__.py:4-5`;
  no new imports are needed in `__init__.py`.
- The `.venv` imports `polar_llama` directly from the repo tree (verified:
  `polar_llama.__file__` resolves to the repo path), so pure-Python edits are
  picked up without re-running `maturin develop`.

## Commands you will need

Run all commands from the repo root, `/Users/daviddrummond/SideProjects/polar-llama`.

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Venv (only if `.venv` missing) | `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt maturin` | exit 0 |
| Build plugin (only if the import below fails) | `source .venv/bin/activate && maturin develop` | exit 0, ends with `Installed polar-llama-...` |
| Sanity import | `.venv/bin/python -c "import polar_llama, polars; print(polars.__version__)"` | prints a version (1.35.1 at planning time), exit 0 |
| New tests only | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests/test_dict_fields.py -v` | all pass (after Step 3) |
| Filtered gate | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests -m "not local_gpu" -k "taxonomy or dict" -q` | 0 failed |
| Regression guard | `.venv/bin/python -m pytest tests/test_optional_fields.py tests/test_structured_outputs.py -q` | 0 failed |
| CI-safe suite | `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py --ignore=tests/test_local_ci_smoke.py -q` | 0 failed |

No live API key is required by any verification in this plan; the `env -u`
prefixes make that explicit.

Baseline facts verified at planning time (before any change):

- `pytest tests -m "not local_gpu" -k "taxonomy or dict" -q` → `9 passed`.
- The CI-safe suite WITHOUT `--ignore=tests/test_local_ci_smoke.py` has 3
  pre-existing failures on dev machines where `mlx` is installed
  (`test_local_ci_smoke.py` asserts mlx is absent — a CI-only assumption).
  They are unrelated to this plan; that is why the ignore flag is in the
  command above. Do not try to fix them.

## Scope

**In scope** (the only files you may modify/create):
- `polar_llama/__init__.py` — add two helper functions, change the body of
  `_parse_json_to_struct`, update the two docstring blocks quoted above.
- `tests/test_dict_fields.py` — create (new keyless test file).

**Out of scope** (do NOT touch, even though they look related):
- `tests/test_taxonomy_tagging.py` — live-API tests; their skip hygiene is
  plans/002-ci-verification-gates.md's business.
- `_field_schema_to_polars_dtype` / `_json_schema_to_polars_dtype` — the
  dict→Utf8 dtype mapping is deliberately KEPT (option A). Do not change the
  derived dtypes; a `List(Struct{key,value})` representation (option B) was
  considered and rejected as heavier and schema-breaking for existing users.
- The Rust layer (`src/`), `polar_llama/optimize.py`, `polar_llama/local/`.
- README or docs beyond the two docstring blocks listed.
- Any live-API testing.

## Git workflow

- Branch: `advisor/017-taxonomy-thinking-field` (branched from `main`)
- Commit style: sentence-case imperative summary, e.g.
  `Preserve dict-typed structured-output fields as JSON strings`
  (matches history such as "Refresh cargo-audit ignore rationale for the pyo3 CVEs")
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Environment sanity

`cd /Users/daviddrummond/SideProjects/polar-llama && .venv/bin/python -c "import polar_llama; print(polar_llama.__file__)"`
→ prints `/Users/daviddrummond/SideProjects/polar-llama/polar_llama/__init__.py`.
If the import fails with a missing extension module, run
`source .venv/bin/activate && maturin develop` once, then retry. If
`polar_llama.__file__` resolves to a `site-packages` path instead of the repo,
STOP (edits to the repo file would not be exercised by the tests).

Create the branch: `git checkout -b advisor/017-taxonomy-thinking-field`.

**Verify**: `git branch --show-current` → `advisor/017-taxonomy-thinking-field`.

### Step 1: Write the failing characterization tests

Create `tests/test_dict_fields.py` with exactly this content:

```python
"""Keyless tests for Dict-typed fields in structured outputs.

Regression tests for the bug where a Pydantic Dict[str, str] field (mapped to
pl.Utf8 by _field_schema_to_polars_dtype) was silently mangled or nulled by
_parse_json_to_struct, because the LLM returns a JSON object for it and a JSON
object cannot land in a Utf8 struct slot. The fix serializes such values to
JSON strings before decoding, so they round-trip through json.loads.

No API keys, no HTTP calls, no LLM — everything is driven through the internal
schema/parsing helpers, following the pattern of tests/test_optional_fields.py.
"""
import json
from typing import Dict, List, Optional

import polars as pl
import pytest
from pydantic import BaseModel

from polar_llama import (
    _create_taxonomy_pydantic_model,
    _json_schema_to_polars_dtype,
    _parse_json_to_struct,
    _pydantic_to_json_schema,
)

TAXONOMY = {
    "sentiment": {
        "description": "The emotional tone of the text",
        "values": {
            "positive": "Text expresses positive emotions",
            "negative": "Text expresses negative emotions",
        },
    }
}


def _taxonomy_dtype():
    model = _create_taxonomy_pydantic_model(TAXONOMY)
    schema = _pydantic_to_json_schema(model)
    return _json_schema_to_polars_dtype(schema)


def test_taxonomy_dict_field_dtype_is_utf8():
    """Dict[str, str] fields are represented as Utf8 (JSON string) columns."""
    dtype = _taxonomy_dtype()
    sentiment_dtype = {f.name: f.dtype for f in dtype.fields}["sentiment"]
    field_dict = {f.name: f.dtype for f in sentiment_dtype.fields}
    assert field_dict["thinking"] == pl.Utf8
    assert field_dict["reflection"] == pl.Utf8
    assert field_dict["value"] == pl.Utf8
    assert field_dict["confidence"] == pl.Float64


def test_taxonomy_thinking_dict_round_trips():
    """Characterization of the original bug: the per-value reasoning dict
    returned by the LLM must survive parsing as a faithful JSON string.

    Before the fix, json_decode(dtype=...) raised on the object-into-Utf8
    mismatch and the cast fallback produced either a mangled keys-dropped
    string (polars 1.35.x) or null (other versions).
    """
    dtype = _taxonomy_dtype()
    thinking_0 = {"positive": "yes it is upbeat", "negative": "no anger present"}
    thinking_1 = {"positive": "no praise present", "negative": "clear frustration"}
    responses = pl.Series([
        json.dumps({"sentiment": {"thinking": thinking_0, "reflection": "clearly positive", "value": "positive", "confidence": 0.9}}),
        json.dumps({"sentiment": {"thinking": thinking_1, "reflection": "clearly negative", "value": "negative", "confidence": 0.8}}),
    ])

    result = _parse_json_to_struct(responses, dtype)
    sentiment = result.struct.field("sentiment")

    thinking = sentiment.struct.field("thinking")
    assert thinking[0] is not None
    assert json.loads(thinking[0]) == thinking_0
    assert thinking[1] is not None
    assert json.loads(thinking[1]) == thinking_1

    # The fields that always survived must still survive.
    assert sentiment.struct.field("value").to_list() == ["positive", "negative"]
    assert sentiment.struct.field("reflection").to_list() == ["clearly positive", "clearly negative"]
    assert sentiment.struct.field("confidence").to_list() == [0.9, 0.8]


class UserModelWithDict(BaseModel):
    """Any user model with a Dict field must benefit, not just the taxonomy."""
    name: str
    attributes: Dict[str, str]


def test_user_model_dict_field_round_trips():
    schema = _pydantic_to_json_schema(UserModelWithDict)
    dtype = _json_schema_to_polars_dtype(schema)
    assert {f.name: f.dtype for f in dtype.fields}["attributes"] == pl.Utf8

    attrs = {"color": "red", "size": "large"}
    responses = pl.Series([json.dumps({"name": "widget", "attributes": attrs})])
    result = _parse_json_to_struct(responses, dtype)
    assert result.struct.field("name")[0] == "widget"
    assert json.loads(result.struct.field("attributes")[0]) == attrs


class ModelWithOptionalDict(BaseModel):
    name: str
    meta: Optional[Dict[str, str]] = None


def test_optional_dict_field_round_trips_and_nulls():
    schema = _pydantic_to_json_schema(ModelWithOptionalDict)
    dtype = _json_schema_to_polars_dtype(schema)

    meta = {"k": "v"}
    responses = pl.Series([
        json.dumps({"name": "a", "meta": meta}),
        json.dumps({"name": "b", "meta": None}),
    ])
    result = _parse_json_to_struct(responses, dtype)
    assert json.loads(result.struct.field("meta")[0]) == meta
    assert result.struct.field("meta")[1] is None


class ModelWithListOfDicts(BaseModel):
    items: List[Dict[str, str]]


def test_list_of_dicts_field_round_trips():
    schema = _pydantic_to_json_schema(ModelWithListOfDicts)
    dtype = _json_schema_to_polars_dtype(schema)
    assert {f.name: f.dtype for f in dtype.fields}["items"] == pl.List(pl.Utf8)

    items = [{"a": "1"}, {"b": "2"}]
    responses = pl.Series([json.dumps({"items": items})])
    result = _parse_json_to_struct(responses, dtype)
    parsed = [json.loads(v) for v in result.struct.field("items")[0].to_list()]
    assert parsed == items


def test_dict_field_null_row_stays_null():
    """A wholly-null response row must not break normalization."""
    dtype = _taxonomy_dtype()
    thinking = {"positive": "p", "negative": "n"}
    responses = pl.Series([
        json.dumps({"sentiment": {"thinking": thinking, "reflection": "r", "value": "positive", "confidence": 1.0}}),
        None,
    ])
    result = _parse_json_to_struct(responses, dtype)
    assert json.loads(result.struct.field("sentiment").struct.field("thinking")[0]) == thinking
    assert result.struct.field("sentiment")[1] is None or all(
        v is None for v in result.struct.field("sentiment")[1].values()
    )


class PlainDictFreeModel(BaseModel):
    is_claim: bool
    reworded_text: Optional[str] = None
    reasoning: str


def test_dict_free_models_unaffected():
    """Models without Dict fields must parse exactly as before the fix."""
    schema = _pydantic_to_json_schema(PlainDictFreeModel)
    dtype = _json_schema_to_polars_dtype(schema)
    responses = pl.Series([
        '{"is_claim": true, "reworded_text": null, "reasoning": "Simple statement"}',
        '{"is_claim": false, "reworded_text": "Reworded", "reasoning": "Needs work"}',
    ])
    result = _parse_json_to_struct(responses, dtype)
    assert result.struct.field("is_claim").to_list() == [True, False]
    assert result.struct.field("reworded_text").to_list() == [None, "Reworded"]
    assert result.struct.field("reasoning").to_list() == ["Simple statement", "Needs work"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests/test_dict_fields.py -v`
→ `5 failed, 2 passed` (confirmed against polars 1.35.1 at planning time).
Expected failing tests:
`test_taxonomy_thinking_dict_round_trips`,
`test_user_model_dict_field_round_trips`,
`test_optional_dict_field_round_trips_and_nulls`, and
`test_dict_field_null_row_stays_null` — each with
`json.decoder.JSONDecodeError` (mangled string) or an assertion on a null/
wrong value, depending on the installed polars — plus
`test_list_of_dicts_field_round_trips` (TypeError/None on 1.35.1; may vary
by polars version).
`test_taxonomy_dict_field_dtype_is_utf8` and
`test_dict_free_models_unaffected` must PASS already.
If `test_taxonomy_thinking_dict_round_trips` passes before any fix, STOP
(the premise is wrong for this environment).

Commit: `Add failing characterization tests for dict-typed structured-output fields`

### Step 2: Implement dtype-driven JSON normalization in `polar_llama/__init__.py`

Insert the following two module-level functions immediately BEFORE
`_parse_json_to_struct` (i.e. above current line 195; after the end of
`_field_schema_to_polars_dtype`). This exact code was validated against
polars 1.35.1 during planning:

```python
def _normalize_json_value_for_dtype(value, dtype):
    """Recursively align a parsed JSON value with the target Polars dtype.

    Dict-typed Pydantic fields (e.g. Dict[str, str]) are mapped to pl.Utf8 by
    _field_schema_to_polars_dtype, but the LLM returns a JSON object for them.
    json_decode cannot put an object into a Utf8 slot (it raises, or mangles/
    nulls the value via the cast fallback, depending on the polars version),
    so any dict or list found where the dtype expects a string is re-serialized
    to its JSON text here. Users get a JSON string they can json.loads() or
    .str.json_decode() themselves.
    """
    if value is None:
        return None
    if dtype == pl.Utf8:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return value
    if isinstance(dtype, pl.Struct) and isinstance(value, dict):
        field_dtypes = {f.name: f.dtype for f in dtype.fields}
        return {
            k: _normalize_json_value_for_dtype(v, field_dtypes[k]) if k in field_dtypes else v
            for k, v in value.items()
        }
    if isinstance(dtype, pl.List) and isinstance(value, list):
        return [_normalize_json_value_for_dtype(item, dtype.inner) for item in value]
    return value


def _normalize_json_str_for_dtype(json_str: Optional[str], dtype: pl.DataType) -> Optional[str]:
    """Normalize one raw JSON response string against the target dtype.

    Rows that are None or not valid JSON are returned unchanged so the
    existing fallback paths in _parse_json_to_struct keep their behavior.
    """
    if json_str is None:
        return None
    try:
        obj = json.loads(json_str)
    except (ValueError, TypeError):
        return json_str
    return json.dumps(_normalize_json_value_for_dtype(obj, dtype))
```

Then replace the body of `_parse_json_to_struct` (currently lines 195-215,
quoted in full in "Current state") with:

```python
def _parse_json_to_struct(json_str_series: pl.Series, dtype: pl.DataType) -> pl.Series:
    """Parse a JSON string series into a struct series.

    The dtype parameter is critical - it ensures Polars uses the schema derived
    from the Pydantic model rather than inferring from data. This fixes issues
    with Optional fields that may be null in some rows but present in others.

    Before decoding, each row is normalized against the dtype: values that the
    schema maps to Utf8 but that arrive as JSON objects/arrays (e.g. Pydantic
    Dict[str, str] fields such as tag_taxonomy's per-field "thinking") are
    re-serialized to JSON strings, so they survive decoding instead of being
    mangled or nulled.
    """
    normalized = pl.Series(
        json_str_series.name,
        [_normalize_json_str_for_dtype(v, dtype) for v in json_str_series],
        dtype=pl.Utf8,
    )
    # Pass the dtype to json_decode to use the Pydantic-derived schema
    # instead of inferring from data. This ensures Optional fields are
    # always included even when null in early rows.
    try:
        return normalized.str.json_decode(dtype=dtype)
    except Exception:
        # If parsing fails with schema, try without and cast
        # This handles edge cases where the response doesn't match schema
        try:
            parsed = normalized.str.json_decode()
            return parsed.cast(dtype, strict=False)
        except Exception:
            # Last resort: return inferred schema
            return normalized.str.json_decode()
```

Notes:
- `json` and `Optional` are already imported at the top of the file
  (lines 4-5); add no imports.
- Do NOT change `_field_schema_to_polars_dtype`, the three
  `map_batches` call sites, or any function signature.
- Use `pl.Utf8` (the codebase convention), not `pl.String`.

**Verify**: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests/test_dict_fields.py -v`
→ `7 passed`.

### Step 3: Regression guard on the existing parsing tests

**Verify**: `.venv/bin/python -m pytest tests/test_optional_fields.py tests/test_structured_outputs.py -q`
→ 0 failed (baseline at planning time: `test_optional_fields.py` alone was
`7 passed`).

Commit: `Preserve dict-typed structured-output fields as JSON strings`
(include the `__init__.py` change and, if not committed in Step 1, the test file).

### Step 4: Update the two docstring blocks to document the JSON-string representation

In `polar_llama/__init__.py`:

1. Replace the block at lines 239-243 (inside `_create_taxonomy_pydantic_model`'s
   docstring; exact current text quoted in "Current state") with:

   ```
       Each field in the output model will be a struct containing:
       - thinking: Dict[str, str] - reasoning for each possible value.
         Returned in the DataFrame as a JSON string (parse with json.loads
         or .str.json_decode); Polars structs cannot hold arbitrary dicts.
       - reflection: str - overall reflection on the field analysis
       - value: str - the selected value
       - confidence: float - confidence in the selection (0.0 to 1.0)
   ```

2. Replace the block at lines 1041-1047 (the "Returns" section of
   `tag_taxonomy`'s docstring; exact current text quoted in "Current state")
   with:

   ```
       polars.Expr
           Expression with structured tags as a Struct. Each taxonomy field becomes
           a nested struct containing:
           - thinking: reasoning for each possible value, as a JSON string
             mapping value name -> reasoning (parse with json.loads or
             .str.json_decode)
           - reflection: str - overall reflection on the field analysis
           - value: str - the selected value
           - confidence: float - confidence score (0.0 to 1.0)
   ```

Line numbers will have shifted by the Step 2 insertion (~40 lines); locate the
blocks by their quoted text, not by line number.

**Verify**:
`.venv/bin/python -c "import polar_llama; d = polar_llama.tag_taxonomy.__doc__; assert 'JSON string' in d, 'docstring not updated'"`
→ exit 0, no output.

Commit: `Document JSON-string representation of taxonomy thinking field`

### Step 5: Full verification sweep

Run, in order:

1. `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests/test_dict_fields.py -v` → `7 passed`
2. `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests -m "not local_gpu" -k "taxonomy or dict" -q` → `16 passed` (9 pre-existing + 7 new), 0 failed
3. `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py --ignore=tests/test_local_ci_smoke.py -q` → 0 failed (skips are fine)
4. `git status --short` → only `polar_llama/__init__.py` modified and `tests/test_dict_fields.py` added (plus `plans/README.md` if you update the index)

**Verify**: all four commands succeed as stated.

## Test plan

- New keyless test file `tests/test_dict_fields.py` (full content in Step 1),
  modeled structurally on `tests/test_optional_fields.py`:
  - `test_taxonomy_thinking_dict_round_trips` — THE regression test for this
    bug: taxonomy-derived dtype + hand-built LLM response with dict-valued
    `thinking`; asserts `json.loads` round-trip of `thinking` AND that
    `value`/`reflection`/`confidence` still parse. Fails before Step 2,
    passes after.
  - `test_user_model_dict_field_round_trips` — proves the fix is generic to
    any user Pydantic model with `Dict[str, str]`, not a taxonomy
    special-case.
  - `test_optional_dict_field_round_trips_and_nulls` — `Optional[Dict[str, str]]`
    (the anyOf schema path) with a null row.
  - `test_list_of_dicts_field_round_trips` — `List[Dict[str, str]]` maps to
    `List(Utf8)` with each element a JSON string.
  - `test_dict_field_null_row_stays_null` — a null response row passes through
    normalization unharmed.
  - `test_taxonomy_dict_field_dtype_is_utf8` — pins the representation choice
    (dtype stays Utf8).
  - `test_dict_free_models_unaffected` — guard that Dict-free models parse
    exactly as before.
- Existing tests that must keep passing: `tests/test_optional_fields.py` (7),
  `tests/test_structured_outputs.py`, and the live-API
  `tests/test_taxonomy_tagging.py` tests (which self-skip keyless — 3
  collected, 3 passed-as-noop without keys).
- Verification: commands in Step 5.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests/test_dict_fields.py -v`
      → `7 passed`, including `test_taxonomy_thinking_dict_round_trips`
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests -m "not local_gpu" -k "taxonomy or dict" -q`
      → 0 failed (16 passed expected)
- [ ] `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY .venv/bin/python -m pytest tests -m "not local_gpu" --ignore=tests/test_parallel_inference.py --ignore=tests/test_local_ci_smoke.py -q`
      → 0 failed
- [ ] `grep -n "_normalize_json_value_for_dtype" polar_llama/__init__.py`
      → at least 2 matches (definition + use)
- [ ] `grep -c "additionalProperties" polar_llama/__init__.py` → unchanged
      from baseline (the dtype mapping was not touched); baseline at
      planning time: 6
- [ ] `.venv/bin/python -c "import polar_llama; assert 'JSON string' in polar_llama.tag_taxonomy.__doc__"`
      → exit 0
- [ ] `git status --short` shows no modified files outside
      `polar_llama/__init__.py`, `tests/test_dict_fields.py`, and
      `plans/README.md`
- [ ] `plans/README.md` status row updated (unless a reviewer told you they
      maintain the index)

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `polar_llama/__init__.py` changed since `afc78da` and
  the "Current state" excerpts (dict→Utf8 branch, `_parse_json_to_struct`
  body, taxonomy `thinking=(Dict[str, str], ...)` field) no longer match —
  especially if a plans/018-*.md executor already modified
  `_field_schema_to_polars_dtype` or `_parse_json_to_struct`.
- In Step 1, `test_taxonomy_thinking_dict_round_trips` PASSES before any fix —
  the installed polars already handles object-into-Utf8 losslessly and the
  premise no longer holds. Report the installed polars version and the
  observed `thinking` value.
- In Step 1, the observed failure mode is neither a mangled string nor a
  null/assertion failure (e.g. `_parse_json_to_struct` raises out of all three
  fallback paths on well-formed input) — report the exact exception; the fix
  shape depends on `json_decode`'s behavior in the installed polars.
- After Step 2, `test_dict_free_models_unaffected` or any test in
  `tests/test_optional_fields.py` / `tests/test_structured_outputs.py` fails —
  the normalization changed behavior for Dict-free models, which must not
  happen.
- `polar_llama.__file__` resolves outside the repo tree (Step 0) — the venv is
  not importing the source you are editing.
- Any step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. the Rust
  layer, or changing the dtype mapping itself).

## Maintenance notes

- **Interaction with plan 018**: a later plan modifies the same functions
  (`_field_schema_to_polars_dtype` / `_parse_json_to_struct`). This plan was
  written to land first; whoever executes 018 must re-verify its excerpts
  against the post-017 code.
- **Representation contract**: dict-typed fields now come back as JSON *strings*
  (users call `json.loads` or `.str.json_decode` on them). This is observable
  API surface documented in the `tag_taxonomy` docstring — if a future polars
  gains a first-class map/variant type, revisiting option B
  (`List(Struct{key,value})` or a native map) is a deliberate follow-up, and a
  breaking change for anyone parsing the JSON strings.
- **Performance**: `_parse_json_to_struct` now does a per-row Python
  `json.loads`/`json.dumps` round-trip before decoding. Response series are
  LLM outputs (network-bound, small n), so this is negligible today; if
  someone ever routes very large non-LLM series through this function, the
  normalization pass is the place to optimize (e.g. skip it when a schema
  walk proves no Utf8 slot can receive a container).
- **Reviewer should scrutinize**: (1) that unknown JSON-schema types (the
  `else: return pl.Utf8` branch at the end of `_field_schema_to_polars_dtype`)
  now also benefit — an object landing in such a field becomes a JSON string
  rather than being lost, which is intended; (2) that keys present in the
  response but absent from the dtype pass through `_normalize_json_value_for_dtype`
  unchanged (the `if k in field_dtypes else v` arm) — same as pre-fix behavior;
  (3) the `except (ValueError, TypeError)` in `_normalize_json_str_for_dtype`
  deliberately swallows only JSON-parse failures so invalid rows keep today's
  fallback behavior.
- **Deliberately deferred**: pytest.skip hygiene for
  `tests/test_taxonomy_tagging.py` (plans/002); a live-API end-to-end
  assertion that a real provider's taxonomy response round-trips (belongs to
  the manual `test-llm-apis.yml` workflow, never to keyless CI).
