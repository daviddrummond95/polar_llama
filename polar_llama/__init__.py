from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Type
import json

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr
    from pydantic import BaseModel

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

# Import Provider enum directly from the extension module
try:
    # First try relative import from the extension module in current directory
    from .polar_llama import Provider
except ImportError:
    # Fallback to try absolute import
    try:
        from polar_llama.polar_llama import Provider
    except ImportError:
        # Define a basic Provider class as fallback if neither import works
        class Provider:
            OPENAI = "openai"
            ANTHROPIC = "anthropic"
            GEMINI = "gemini"
            GROQ = "groq"
            BEDROCK = "bedrock"
            
            def __init__(self, provider_str):
                self.value = provider_str
                
            def __str__(self):
                return self.value

# Import and initialize the expressions helper to ensure expressions are registered
from polar_llama.expressions import ensure_expressions_registered, get_lib_path

# Ensure the expressions are registered
ensure_expressions_registered()
# Update the lib path to make sure we're using the actual library
lib = get_lib_path()

def _pydantic_to_json_schema(model: Type['BaseModel']) -> dict:
    """Convert a Pydantic model to JSON schema."""
    try:
        from pydantic import BaseModel
        if not issubclass(model, BaseModel):
            raise ValueError("response_model must be a Pydantic BaseModel subclass")

        # Get the JSON schema from the Pydantic model
        schema = model.model_json_schema()

        # Recursively add additionalProperties: false to all objects
        # This is required by some providers like Groq
        def add_additional_properties_false(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    obj["additionalProperties"] = False
                # Recursively process nested objects
                for key, value in obj.items():
                    if isinstance(value, dict):
                        add_additional_properties_false(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add_additional_properties_false(item)

        add_additional_properties_false(schema)
        return schema
    except ImportError:
        raise ImportError("Pydantic is required for structured outputs. Install with: pip install pydantic>=2.0.0")

def _json_schema_to_polars_dtype(schema: dict) -> pl.DataType:
    """Convert a JSON schema to a Polars DataType."""
    properties = schema.get("properties", {})

    fields = []
    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type")

        # Map JSON schema types to Polars types
        if field_type == "string":
            pl_type = pl.Utf8
        elif field_type == "integer":
            pl_type = pl.Int64
        elif field_type == "number":
            pl_type = pl.Float64
        elif field_type == "boolean":
            pl_type = pl.Boolean
        elif field_type == "array":
            # Handle arrays - use List type
            items_schema = field_schema.get("items", {})
            items_type = items_schema.get("type", "string")
            if items_type == "string":
                pl_type = pl.List(pl.Utf8)
            elif items_type == "integer":
                pl_type = pl.List(pl.Int64)
            elif items_type == "number":
                pl_type = pl.List(pl.Float64)
            else:
                pl_type = pl.List(pl.Utf8)  # Default to string list
        elif field_type == "object":
            # Nested objects - recursively convert
            pl_type = _json_schema_to_polars_dtype(field_schema)
        else:
            # Default to string for unknown types
            pl_type = pl.Utf8

        fields.append(pl.Field(field_name, pl_type))

    # Add error fields for error handling
    fields.append(pl.Field("_error", pl.Utf8))
    fields.append(pl.Field("_details", pl.Utf8))
    fields.append(pl.Field("_raw", pl.Utf8))

    return pl.Struct(fields)

def _parse_json_to_struct(json_str_series: pl.Series, dtype: pl.DataType) -> pl.Series:
    """Parse a JSON string series into a struct series."""
    # Use Polars' built-in JSON parsing
    try:
        # First, try to decode as JSON
        parsed = json_str_series.str.json_decode()

        # Cast to the target struct type to ensure field types match
        return parsed.cast(dtype, strict=False)
    except Exception:
        # If parsing fails, return the series as-is (will contain error objects)
        return json_str_series.str.json_decode()

def inference_async(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type['BaseModel']] = None
) -> pl.Expr:
    """
    Asynchronously infer completions for the given text expressions using an LLM.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
    response_model : Type[BaseModel], optional
        A Pydantic model class to define structured output schema.
        The LLM response will be validated against this schema.
        Returns a Struct with fields matching the Pydantic model.

    Returns
    -------
    polars.Expr
        Expression with inferred completions as a Struct (if response_model provided)
        or String (if no response_model)
    """
    expr = parse_into_expr(expr)
    kwargs = {}

    if provider is not None:
        # Convert Provider to string to make it picklable
        if isinstance(provider, Provider):
            provider = str(provider)
        kwargs["provider"] = provider

    if model is not None:
        kwargs["model"] = model

    struct_dtype = None
    if response_model is not None:
        schema = _pydantic_to_json_schema(response_model)
        # Pass the JSON schema as a JSON string to Rust
        kwargs["response_schema"] = json.dumps(schema)
        kwargs["response_model_name"] = response_model.__name__
        # Create the target struct dtype for later conversion
        struct_dtype = _json_schema_to_polars_dtype(schema)

    result_expr = register_plugin(
        args=[expr],
        symbol="inference_async",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )

    # If response_model was provided, convert JSON strings to structs
    if struct_dtype is not None:
        # Use map_batches to convert the JSON string series to struct series
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype),
            return_dtype=struct_dtype
        )

    return result_expr

def inference(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type['BaseModel']] = None
) -> pl.Expr:
    """
    Synchronously infer completions for the given text expressions using an LLM.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
    response_model : Type[BaseModel], optional
        A Pydantic model class to define structured output schema.
        The LLM response will be validated against this schema.
        Returns a Struct with fields matching the Pydantic model.

    Returns
    -------
    polars.Expr
        Expression with inferred completions as a Struct (if response_model provided)
        or String (if no response_model)
    """
    expr = parse_into_expr(expr)
    kwargs = {}

    if provider is not None:
        # Convert Provider to string to make it picklable
        if isinstance(provider, Provider):
            provider = str(provider)
        kwargs["provider"] = provider

    if model is not None:
        kwargs["model"] = model

    struct_dtype = None
    if response_model is not None:
        schema = _pydantic_to_json_schema(response_model)
        # Pass the JSON schema as a JSON string to Rust
        kwargs["response_schema"] = json.dumps(schema)
        kwargs["response_model_name"] = response_model.__name__
        # Create the target struct dtype for later conversion
        struct_dtype = _json_schema_to_polars_dtype(schema)

    result_expr = register_plugin(
        args=[expr],
        symbol="inference",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )

    # If response_model was provided, convert JSON strings to structs
    if struct_dtype is not None:
        # Use map_batches to convert the JSON string series to struct series
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype),
            return_dtype=struct_dtype
        )

    return result_expr

def inference_messages(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type['BaseModel']] = None
) -> pl.Expr:
    """
    Process message arrays (conversations) for inference using LLMs.

    This function accepts properly formatted JSON message arrays and sends them
    to the LLM for inference while preserving conversation context.

    Parameters
    ----------
    expr : polars.Expr
        The expression containing JSON message arrays
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
    response_model : Type[BaseModel], optional
        A Pydantic model class to define structured output schema.
        The LLM response will be validated against this schema.
        Returns a Struct with fields matching the Pydantic model.

    Returns
    -------
    polars.Expr
        Expression with inferred completions as a Struct (if response_model provided)
        or String (if no response_model)
    """
    expr = parse_into_expr(expr)
    kwargs = {}

    if provider is not None:
        # Convert Provider to string to make it picklable
        if hasattr(provider, 'as_str'):
            provider_str = provider.as_str()
        elif hasattr(provider, '__str__'):
            provider_str = str(provider)
        else:
            provider_str = provider

        kwargs["provider"] = provider_str

    if model is not None:
        kwargs["model"] = model

    struct_dtype = None
    if response_model is not None:
        schema = _pydantic_to_json_schema(response_model)
        # Pass the JSON schema as a JSON string to Rust
        kwargs["response_schema"] = json.dumps(schema)
        kwargs["response_model_name"] = response_model.__name__
        # Create the target struct dtype for later conversion
        struct_dtype = _json_schema_to_polars_dtype(schema)

    # Don't pass empty kwargs dictionary
    if not kwargs:
        result_expr = register_plugin(
            args=[expr],
            symbol="inference_messages",
            is_elementwise=True,
            lib=lib,
        )
    else:
        result_expr = register_plugin(
            args=[expr],
            symbol="inference_messages",
            is_elementwise=True,
            lib=lib,
            kwargs=kwargs,
        )

    # If response_model was provided, convert JSON strings to structs
    if struct_dtype is not None:
        # Use map_batches to convert the JSON string series to struct series
        result_expr = result_expr.map_batches(
            lambda s: _parse_json_to_struct(s, struct_dtype),
            return_dtype=struct_dtype
        )

    return result_expr

def string_to_message(expr: IntoExpr, *, message_type: str) -> pl.Expr:
    """
    Convert a string to a message with the specified type.
    
    Parameters
    ----------
    expr : polars.Expr
        The text expression to convert
    message_type : str
        The type of message to create ("user", "system", "assistant")
        
    Returns
    -------
    polars.Expr
        Expression with formatted messages
    """
    expr = parse_into_expr(expr)
    return register_plugin(
        args=[expr],
        symbol="string_to_message",
        is_elementwise=True,
        lib=lib,
        kwargs={"message_type": message_type},
    )

def combine_messages(*exprs: IntoExpr) -> pl.Expr:
    """
    Combine multiple message expressions into a single message array.
    
    This function takes multiple message expressions (strings containing JSON formatted messages)
    and combines them into a single JSON array of messages, preserving the order.
    
    Parameters
    ----------
    *exprs : polars.Expr
        One or more expressions containing messages to combine
        
    Returns
    -------
    polars.Expr
        Expression with combined message arrays
    """
    args = [parse_into_expr(expr) for expr in exprs]
    
    return register_plugin(
        args=args,
        symbol="combine_messages",
        is_elementwise=True,
        lib=lib,
    )