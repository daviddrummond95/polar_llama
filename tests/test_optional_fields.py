"""Tests for Optional field handling in schema conversion."""
import pytest
import polars as pl
from pydantic import BaseModel
from typing import Optional, List

# Import internal functions for testing
from polar_llama import _pydantic_to_json_schema, _json_schema_to_polars_dtype, _parse_json_to_struct


class ModelWithOptionalFields(BaseModel):
    """Test model with various Optional field types."""
    required_field: str
    optional_string: Optional[str] = None
    optional_int: Optional[int] = None
    optional_float: Optional[float] = None
    optional_bool: Optional[bool] = None


class ModelWithOptionalList(BaseModel):
    """Test model with Optional list field."""
    name: str
    tags: Optional[List[str]] = None


class NestedModel(BaseModel):
    """Nested model for testing."""
    value: str
    count: int


class ModelWithOptionalNested(BaseModel):
    """Test model with Optional nested object."""
    title: str
    nested: Optional[NestedModel] = None


class ClaimClassification(BaseModel):
    """Model from the bug report."""
    is_claim: bool
    needs_reword: bool
    reworded_text: Optional[str] = None
    reasoning: str


def test_optional_string_schema_conversion():
    """Test that Optional[str] is properly converted to Polars schema."""
    schema = _pydantic_to_json_schema(ModelWithOptionalFields)
    dtype = _json_schema_to_polars_dtype(schema)

    # Check that all fields are present in the struct
    field_names = [f.name for f in dtype.fields]
    assert "required_field" in field_names
    assert "optional_string" in field_names
    assert "optional_int" in field_names
    assert "optional_float" in field_names
    assert "optional_bool" in field_names

    # Check field types
    field_dict = {f.name: f.dtype for f in dtype.fields}
    assert field_dict["required_field"] == pl.Utf8
    assert field_dict["optional_string"] == pl.Utf8
    assert field_dict["optional_int"] == pl.Int64
    assert field_dict["optional_float"] == pl.Float64
    assert field_dict["optional_bool"] == pl.Boolean


def test_optional_list_schema_conversion():
    """Test that Optional[List[str]] is properly converted."""
    schema = _pydantic_to_json_schema(ModelWithOptionalList)
    dtype = _json_schema_to_polars_dtype(schema)

    field_dict = {f.name: f.dtype for f in dtype.fields}
    assert field_dict["name"] == pl.Utf8
    assert field_dict["tags"] == pl.List(pl.Utf8)


def test_claim_classification_schema():
    """Test the exact model from the bug report."""
    schema = _pydantic_to_json_schema(ClaimClassification)
    dtype = _json_schema_to_polars_dtype(schema)

    field_names = [f.name for f in dtype.fields]
    assert "is_claim" in field_names
    assert "needs_reword" in field_names
    assert "reworded_text" in field_names
    assert "reasoning" in field_names

    field_dict = {f.name: f.dtype for f in dtype.fields}
    assert field_dict["is_claim"] == pl.Boolean
    assert field_dict["needs_reword"] == pl.Boolean
    assert field_dict["reworded_text"] == pl.Utf8
    assert field_dict["reasoning"] == pl.Utf8


def test_parse_json_with_optional_null():
    """Test parsing JSON where optional field is null in some rows."""
    schema = _pydantic_to_json_schema(ClaimClassification)
    dtype = _json_schema_to_polars_dtype(schema)

    # Simulate responses where reworded_text is null in first row but present in second
    json_responses = pl.Series([
        '{"is_claim": true, "needs_reword": false, "reworded_text": null, "reasoning": "Simple statement"}',
        '{"is_claim": true, "needs_reword": true, "reworded_text": "Reworded text here", "reasoning": "Needs clarification"}',
    ])

    result = _parse_json_to_struct(json_responses, dtype)

    # Verify the struct has all fields
    assert result.dtype == dtype or isinstance(result.dtype, pl.Struct)

    # Access fields
    reworded = result.struct.field("reworded_text")
    assert reworded[0] is None
    assert reworded[1] == "Reworded text here"

    is_claim = result.struct.field("is_claim")
    assert is_claim[0] == True
    assert is_claim[1] == True


def test_parse_json_with_missing_optional():
    """Test parsing JSON where optional field is completely missing (not just null)."""
    schema = _pydantic_to_json_schema(ClaimClassification)
    dtype = _json_schema_to_polars_dtype(schema)

    # First row has no reworded_text key at all
    json_responses = pl.Series([
        '{"is_claim": true, "needs_reword": false, "reasoning": "No reworded_text key"}',
        '{"is_claim": true, "needs_reword": true, "reworded_text": "Has text", "reasoning": "Has key"}',
    ])

    result = _parse_json_to_struct(json_responses, dtype)

    # Should handle missing keys gracefully
    reworded = result.struct.field("reworded_text")
    assert reworded[0] is None  # Missing key should become null
    assert reworded[1] == "Has text"


def test_parse_json_all_optional_null():
    """Test parsing when all optional fields are null."""
    schema = _pydantic_to_json_schema(ModelWithOptionalFields)
    dtype = _json_schema_to_polars_dtype(schema)

    json_responses = pl.Series([
        '{"required_field": "test1", "optional_string": null, "optional_int": null, "optional_float": null, "optional_bool": null}',
        '{"required_field": "test2", "optional_string": "value", "optional_int": 42, "optional_float": 3.14, "optional_bool": true}',
    ])

    result = _parse_json_to_struct(json_responses, dtype)

    # First row should have all nulls for optional fields
    assert result.struct.field("optional_string")[0] is None
    assert result.struct.field("optional_int")[0] is None
    assert result.struct.field("optional_float")[0] is None
    assert result.struct.field("optional_bool")[0] is None

    # Second row should have values
    assert result.struct.field("optional_string")[1] == "value"
    assert result.struct.field("optional_int")[1] == 42
    assert result.struct.field("optional_float")[1] == 3.14
    assert result.struct.field("optional_bool")[1] == True


def test_error_fields_included():
    """Test that error fields (_error, _details, _raw) are included in schema."""
    schema = _pydantic_to_json_schema(ClaimClassification)
    dtype = _json_schema_to_polars_dtype(schema)

    field_names = [f.name for f in dtype.fields]
    assert "_error" in field_names
    assert "_details" in field_names
    assert "_raw" in field_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
