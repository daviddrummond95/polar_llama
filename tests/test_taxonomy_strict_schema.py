"""Regression tests for issue #51: taxonomy `thinking` field breaking OpenAI strict mode.

These tests build the taxonomy Pydantic model, convert it to a JSON schema, and
parse a sample response -- all locally, with no API key and no mlx dependency.
They must never skip.
"""

import json

import polars as pl

from polar_llama import (
    _create_taxonomy_pydantic_model,
    _pydantic_to_json_schema,
    _json_schema_to_polars_dtype,
    _parse_json_to_struct,
)

# Taxonomy with hostile value names (not valid Python identifiers, contain
# spaces/hyphens/digits) to lock in that the chosen `thinking` shape --
# List[{value, reasoning}] -- never depends on value names being usable as
# schema keys or identifiers.
TAXONOMY = {
    "priority": {
        "description": "How urgent the item is",
        "values": {
            "high-priority": "Needs immediate attention",
            "not applicable": "Priority does not apply",
            "3rd party": "Owned by a third party",
        },
    },
    "sentiment": {
        "description": "Overall tone",
        "values": {
            "positive": "Positive tone",
            "negative": "Negative tone",
        },
    },
}


def _walk_assert_strict_compliant(node):
    """Recursively assert every object node in a JSON schema is OpenAI
    strict-mode compliant: has `properties`, `required` covering every
    property key, `additionalProperties: false`, and never a dynamic map
    (dict-valued additionalProperties)."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert "properties" in node, f"object node missing properties: {node}"
            assert node.get("additionalProperties") is False, (
                f"object node without additionalProperties=false: {node}"
            )
            assert "required" in node and sorted(node["required"]) == sorted(
                node["properties"].keys()
            ), (
                f"required != property keys: {node.get('required')} vs {list(node['properties'])}"
            )

        # No dynamic-key map (dict-valued additionalProperties) may survive
        # anywhere in the schema -- this is the exact shape that caused #51.
        assert not isinstance(node.get("additionalProperties"), dict), (
            f"dynamic map object leaked into schema (issue #51): {node}"
        )

        # OpenAI strict mode forbids a `$ref` node from carrying sibling
        # keywords (e.g. a description). This is the second strict-mode
        # violation surfaced by a live gpt-4o-mini call while fixing #51:
        # "$ref cannot have keywords {'description'}".
        if "$ref" in node:
            assert set(node.keys()) == {"$ref"}, (
                f"$ref node has sibling keywords (rejected by OpenAI strict "
                f"mode): {node}"
            )

        for value in node.values():
            _walk_assert_strict_compliant(value)
    elif isinstance(node, list):
        for item in node:
            _walk_assert_strict_compliant(item)


def test_taxonomy_schema_is_openai_strict_compliant():
    """Would have caught #51: the generated taxonomy schema must not contain
    any dynamic-key map objects, and every object node must be strict-mode
    compliant (properties + required==properties + additionalProperties=false)."""
    model = _create_taxonomy_pydantic_model(TAXONOMY)
    schema = _pydantic_to_json_schema(model)

    _walk_assert_strict_compliant(schema)


def test_taxonomy_dtype_roundtrip():
    """The new `thinking: List[{value, reasoning}]` shape must round-trip
    cleanly into a Polars Struct dtype and parse sample JSON correctly,
    including for taxonomy value names that aren't valid identifiers."""
    model = _create_taxonomy_pydantic_model(TAXONOMY)
    schema = _pydantic_to_json_schema(model)
    dtype = _json_schema_to_polars_dtype(schema)

    field_dtypes = {f.name: f.dtype for f in dtype.fields}
    assert set(field_dtypes.keys()) >= {
        "priority",
        "sentiment",
        "_error",
        "_details",
        "_raw",
    }

    sentiment_dtype = field_dtypes["sentiment"]
    inner = {f.name: f.dtype for f in sentiment_dtype.fields}
    assert inner["thinking"] == pl.List(
        pl.Struct({"value": pl.Utf8, "reasoning": pl.Utf8})
    )
    assert inner["reflection"] == pl.Utf8
    assert inner["value"] == pl.Utf8
    assert inner["confidence"] == pl.Float64

    sample = {
        "sentiment": {
            "thinking": [
                {"value": "positive", "reasoning": "upbeat tone"},
                {"value": "negative", "reasoning": "no negativity present"},
            ],
            "reflection": "clearly positive",
            "value": "positive",
            "confidence": 0.9,
        },
        "priority": {
            "thinking": [
                {"value": "high-priority", "reasoning": "server down"},
                {"value": "not applicable", "reasoning": "n/a for this doc"},
                {"value": "3rd party", "reasoning": "not vendor related"},
            ],
            "reflection": "urgent internal issue",
            "value": "high-priority",
            "confidence": 0.95,
        },
    }

    parsed = _parse_json_to_struct(pl.Series([json.dumps(sample)]), dtype)
    row = parsed[0]

    assert row["sentiment"]["value"] == "positive"
    assert row["sentiment"]["confidence"] == 0.9
    assert row["sentiment"]["thinking"][0] == {
        "value": "positive",
        "reasoning": "upbeat tone",
    }
    assert len(row["priority"]["thinking"]) == 3
    assert row["priority"]["thinking"][0]["value"] == "high-priority"
    assert row["_error"] is None
