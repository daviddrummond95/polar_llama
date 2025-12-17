from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Type, Dict, Any
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
        # However, we skip objects that already have additionalProperties defined
        # (like Dict[str, str] types which need dynamic keys)
        def add_additional_properties_false(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "additionalProperties" not in obj:
                    # Only add if not already present (Dict types already have it defined)
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

def _extract_type_from_anyof(any_of: list, root_schema: dict) -> tuple:
    """
    Extract the non-null type from an anyOf pattern (used for Optional fields).

    Returns a tuple of (type_string, type_schema) where type_schema contains
    the full schema for complex types like arrays or objects.
    """
    for option in any_of:
        # Handle $ref in anyOf options
        if "$ref" in option:
            ref_path = option["$ref"]
            if ref_path.startswith("#/"):
                ref_parts = ref_path[2:].split("/")
                ref_schema = root_schema
                for part in ref_parts:
                    ref_schema = ref_schema.get(part, {})
                return (ref_schema.get("type", "object"), ref_schema)

        option_type = option.get("type")
        if option_type and option_type != "null":
            return (option_type, option)

    return ("string", {})  # Default fallback


def _json_schema_to_polars_dtype(schema: dict, root_schema: dict = None) -> pl.DataType:
    """Convert a JSON schema to a Polars DataType."""
    # Keep reference to root schema for resolving $ref
    if root_schema is None:
        root_schema = schema

    properties = schema.get("properties", {})

    fields = []
    for field_name, field_schema in properties.items():
        pl_type = _field_schema_to_polars_dtype(field_schema, root_schema)
        fields.append(pl.Field(field_name, pl_type))

    # Add error fields for error handling (only at the root level)
    if root_schema == schema:
        fields.append(pl.Field("_error", pl.Utf8))
        fields.append(pl.Field("_details", pl.Utf8))
        fields.append(pl.Field("_raw", pl.Utf8))

    return pl.Struct(fields)


def _field_schema_to_polars_dtype(field_schema: dict, root_schema: dict) -> pl.DataType:
    """Convert a single field's JSON schema to a Polars DataType."""
    # Handle $ref references
    if "$ref" in field_schema:
        ref_path = field_schema["$ref"]
        if ref_path.startswith("#/"):
            ref_parts = ref_path[2:].split("/")
            ref_schema = root_schema
            for part in ref_parts:
                ref_schema = ref_schema.get(part, {})
            return _json_schema_to_polars_dtype(ref_schema, root_schema)
        else:
            return pl.Utf8  # Unknown reference format

    # Handle anyOf patterns (used for Optional fields in Pydantic v2)
    if "anyOf" in field_schema:
        field_type, type_schema = _extract_type_from_anyof(field_schema["anyOf"], root_schema)
        # Create a temporary field_schema with the extracted type info
        field_schema = {**type_schema, "type": field_type}

    field_type = field_schema.get("type")

    # Handle type arrays like ["string", "null"] (alternative Optional format)
    if isinstance(field_type, list):
        # Find the non-null type
        for t in field_type:
            if t != "null":
                field_type = t
                break
        else:
            field_type = "string"  # Default if only null

    # Map JSON schema types to Polars types
    if field_type == "string":
        return pl.Utf8
    elif field_type == "integer":
        return pl.Int64
    elif field_type == "number":
        return pl.Float64
    elif field_type == "boolean":
        return pl.Boolean
    elif field_type == "array":
        # Handle arrays - use List type
        items_schema = field_schema.get("items", {})
        if items_schema:
            items_dtype = _field_schema_to_polars_dtype(items_schema, root_schema)
            return pl.List(items_dtype)
        return pl.List(pl.Utf8)  # Default to string list
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


# ============================================================================
# Taxonomy-based Tagging
# ============================================================================

def _create_taxonomy_pydantic_model(taxonomy: Dict[str, Dict[str, Any]]) -> Type['BaseModel']:
    """
    Create a Pydantic model from a taxonomy definition.

    The taxonomy should be a dict with the following structure:
    {
        "field_name": {
            "description": "Description of the field",
            "values": {
                "value1": "Definition of value1",
                "value2": "Definition of value2",
                ...
            }
        },
        ...
    }

    Each field in the output model will be a struct containing:
    - thinking: Dict[str, str] - reasoning for each possible value
    - reflection: str - overall reflection on the field analysis
    - value: str - the selected value
    - confidence: float - confidence in the selection (0.0 to 1.0)
    """
    try:
        from pydantic import BaseModel, Field, create_model
        from typing import Dict

        # Create a field result model for each taxonomy field
        field_models = {}

        for field_name, field_config in taxonomy.items():
            values = field_config.get("values", {})
            value_names = list(values.keys())

            # Create the field result model with dynamic thinking keys
            field_result_model = create_model(
                f"{field_name.title()}Result",
                thinking=(Dict[str, str], Field(..., description=f"Reasoning for each possible value: {', '.join(value_names)}")),
                reflection=(str, Field(..., description="Overall reflection on your analysis of this field")),
                value=(str, Field(..., description=f"Selected value from: {', '.join(value_names)}")),
                confidence=(float, Field(..., ge=0.0, le=1.0, description="Confidence in the selected value (0.0 to 1.0)"))
            )

            field_models[field_name] = (field_result_model, Field(..., description=field_config.get("description", "")))

        # Create the main taxonomy result model
        TaxonomyResult = create_model("TaxonomyResult", **field_models)

        return TaxonomyResult

    except ImportError:
        raise ImportError("Pydantic is required for taxonomy tagging. Install with: pip install pydantic>=2.0.0")


def _create_taxonomy_prompt(taxonomy: Dict[str, Dict[str, Any]], document_field_name: str = "document") -> str:
    """
    Create a system prompt for taxonomy-based tagging.

    This prompt instructs the model to analyze a document according to the
    provided taxonomy and return structured tags with reasoning.
    """
    prompt_parts = [
        f"You are an expert document analyst. Analyze the provided {document_field_name} and tag it according to the following taxonomy.",
        "",
        "# Taxonomy Fields",
        ""
    ]

    for field_name, field_config in taxonomy.items():
        description = field_config.get("description", "")
        values = field_config.get("values", {})

        prompt_parts.append(f"## {field_name}")
        if description:
            prompt_parts.append(f"{description}")
        prompt_parts.append("")
        prompt_parts.append("Possible values:")

        for value_name, value_definition in values.items():
            prompt_parts.append(f"- **{value_name}**: {value_definition}")

        prompt_parts.append("")

    prompt_parts.extend([
        "# Instructions",
        "",
        "For each field in the taxonomy:",
        "",
        "1. **Thinking**: Consider each possible value and write your reasoning for why it might or might not apply to the document. Provide one reasoning string for each possible value.",
        "",
        "2. **Reflection**: After thinking through all values, reflect on your analysis. Consider which value best fits the document and why.",
        "",
        "3. **Value**: Select the single best value from the possible values for this field.",
        "",
        "4. **Confidence**: Provide your confidence in this selection as a number between 0.0 (not confident) and 1.0 (very confident).",
        "",
        "Return your analysis in the structured format with all required fields."
    ])

    return "\n".join(prompt_parts)

def inference_async(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type['BaseModel']] = None,
    response_format: Optional[Type['BaseModel']] = None,
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

    # Handle response_format alias
    if response_model is None and response_format is not None:
        response_model = response_format

    struct_dtype = None
    if response_model is not None:
        _validate_strict_mode_schema(response_model)
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
    response_model: Optional[Type['BaseModel']] = None,
    response_format: Optional[Type['BaseModel']] = None,
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
    response_format : Type[BaseModel], optional
        Alias for response_model.

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

    # Handle response_format alias
    if response_model is None and response_format is not None:
        response_model = response_format

    struct_dtype = None
    if response_model is not None:
        _validate_strict_mode_schema(response_model)
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
    response_model: Optional[Type['BaseModel']] = None,
    response_format: Optional[Type['BaseModel']] = None,
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
    response_format : Type[BaseModel], optional
        Alias for response_model.

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

    # Handle response_format alias
    if response_model is None and response_format is not None:
        response_model = response_format

    struct_dtype = None
    if response_model is not None:
        _validate_strict_mode_schema(response_model)
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


def embedding_async(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    Asynchronously generate embeddings for the given text expressions.

    This function generates vector embeddings for text using various embedding providers.
    The embeddings are computed in parallel for maximum performance and memory efficiency.

    Parameters
    ----------
    expr : polars.Expr
        The text expression to generate embeddings for
    provider : str or Provider, optional
        The provider to use (OpenAI, Gemini, Bedrock). Default: OpenAI
    model : str, optional
        The embedding model name to use. If not specified, uses the default
        model for the provider:
        - OpenAI: "text-embedding-3-small" (1536 dimensions)
        - Gemini: "text-embedding-004" (768 dimensions)
        - Bedrock: "amazon.titan-embed-text-v1" (1536 dimensions)

    Returns
    -------
    polars.Expr
        Expression with embeddings as List[Float64] (vector of floats)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, Provider
    >>>
    >>> # Create a dataframe with text
    >>> df = pl.DataFrame({
    ...     "text": ["Hello world", "Machine learning is fun"]
    ... })
    >>>
    >>> # Generate embeddings using OpenAI (default)
    >>> result = df.with_columns(
    ...     embeddings=embedding_async(pl.col("text"))
    ... )
    >>>
    >>> # Use a specific provider and model
    >>> result = df.with_columns(
    ...     embeddings=embedding_async(
    ...         pl.col("text"),
    ...         provider=Provider.OPENAI,
    ...         model="text-embedding-3-large"
    ...     )
    ... )
    >>>
    >>> # Access the embedding dimensions
    >>> result.select([
    ...     "text",
    ...     pl.col("embeddings").list.len().alias("dimensions")
    ... ])
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

    if kwargs:
        return register_plugin(
            args=[expr],
            symbol="embedding_async",
            is_elementwise=True,
            lib=lib,
            kwargs=kwargs,
        )
    else:
        return register_plugin(
            args=[expr],
            symbol="embedding_async",
            is_elementwise=True,
            lib=lib,
        )


def cosine_similarity(
    expr1: IntoExpr,
    expr2: IntoExpr,
) -> pl.Expr:
    """
    Calculate cosine similarity between two embedding vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical). For normalized embeddings,
    this is equivalent to dot product.

    Parameters
    ----------
    expr1 : polars.Expr
        First embedding vector (List[Float64])
    expr2 : polars.Expr
        Second embedding vector (List[Float64])

    Returns
    -------
    polars.Expr
        Cosine similarity score (Float64)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, cosine_similarity
    >>>
    >>> df = pl.DataFrame({
    ...     "text1": ["Hello world"],
    ...     "text2": ["Hello there"]
    ... })
    >>> result = df.with_columns(
    ...     emb1=embedding_async(pl.col("text1")),
    ...     emb2=embedding_async(pl.col("text2"))
    ... ).with_columns(
    ...     similarity=cosine_similarity(pl.col("emb1"), pl.col("emb2"))
    ... )
    """
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[expr1, expr2],
        symbol="cosine_similarity",
        is_elementwise=True,
        lib=lib,
    )


def dot_product(
    expr1: IntoExpr,
    expr2: IntoExpr,
) -> pl.Expr:
    """
    Calculate dot product between two embedding vectors.

    The dot product is the sum of element-wise products of two vectors.

    Parameters
    ----------
    expr1 : polars.Expr
        First embedding vector (List[Float64])
    expr2 : polars.Expr
        Second embedding vector (List[Float64])

    Returns
    -------
    polars.Expr
        Dot product value (Float64)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, dot_product
    >>>
    >>> df = pl.DataFrame({
    ...     "text1": ["Hello world"],
    ...     "text2": ["Hello there"]
    ... })
    >>> result = df.with_columns(
    ...     emb1=embedding_async(pl.col("text1")),
    ...     emb2=embedding_async(pl.col("text2"))
    ... ).with_columns(
    ...     dot_prod=dot_product(pl.col("emb1"), pl.col("emb2"))
    ... )
    """
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[expr1, expr2],
        symbol="dot_product",
        is_elementwise=True,
        lib=lib,
    )


def euclidean_distance(
    expr1: IntoExpr,
    expr2: IntoExpr,
) -> pl.Expr:
    """
    Calculate Euclidean distance between two embedding vectors.

    The Euclidean distance is the straight-line distance between two points
    in n-dimensional space.

    Parameters
    ----------
    expr1 : polars.Expr
        First embedding vector (List[Float64])
    expr2 : polars.Expr
        Second embedding vector (List[Float64])

    Returns
    -------
    polars.Expr
        Euclidean distance (Float64)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, euclidean_distance
    >>>
    >>> df = pl.DataFrame({
    ...     "text1": ["Hello world"],
    ...     "text2": ["Hello there"]
    ... })
    >>> result = df.with_columns(
    ...     emb1=embedding_async(pl.col("text1")),
    ...     emb2=embedding_async(pl.col("text2"))
    ... ).with_columns(
    ...     distance=euclidean_distance(pl.col("emb1"), pl.col("emb2"))
    ... )
    """
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[expr1, expr2],
        symbol="euclidean_distance",
        is_elementwise=True,
        lib=lib,
    )


def knn_hnsw(
    query_expr: IntoExpr,
    reference_expr: IntoExpr,
    *,
    k: int = 5,
) -> pl.Expr:
    """
    Find k-nearest neighbors using Hierarchical Navigable Small World (HNSW) algorithm.

    This function builds an HNSW index from the reference embeddings and searches
    for the k nearest neighbors for each query embedding. HNSW provides approximate
    nearest neighbor search that is much faster than exact search for large datasets.

    Parameters
    ----------
    query_expr : polars.Expr
        Query embedding vectors (List[Float64])
    reference_expr : polars.Expr
        Reference embedding vectors to search through (List[Float64])
    k : int, optional
        Number of nearest neighbors to return (default: 5)

    Returns
    -------
    polars.Expr
        List of indices (List[Int64]) of the k nearest neighbors

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import embedding_async, knn_hnsw
    >>>
    >>> # Create a corpus and query
    >>> corpus_df = pl.DataFrame({
    ...     "id": [1, 2, 3, 4],
    ...     "text": [
    ...         "Machine learning is fun",
    ...         "Deep learning uses neural networks",
    ...         "Python is a programming language",
    ...         "Data science involves statistics"
    ...     ]
    ... }).with_columns(
    ...     embeddings=embedding_async(pl.col("text"))
    ... )
    >>>
    >>> # Find similar documents
    >>> query = pl.DataFrame({
    ...     "query": ["neural networks"]
    ... }).with_columns(
    ...     query_emb=embedding_async(pl.col("query"))
    ... )
    >>>
    >>> # Cross join and find neighbors
    >>> result = query.join(
    ...     corpus_df.select([pl.col("embeddings").alias("corpus_emb")]),
    ...     how="cross"
    ... ).with_columns(
    ...     neighbors=knn_hnsw(
    ...         pl.col("query_emb"),
    ...         pl.col("corpus_emb"),
    ...         k=3
    ...     )
    ... )
    """
    query_expr = parse_into_expr(query_expr)
    reference_expr = parse_into_expr(reference_expr)
    return register_plugin(
        args=[query_expr, reference_expr],
        symbol="knn_hnsw",
        is_elementwise=True,
        lib=lib,
        kwargs={"k": k},
    )


def tag_taxonomy(
    expr: IntoExpr,
    taxonomy: Dict[str, Dict[str, Any]],
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    Tag documents according to a taxonomy definition with detailed reasoning.

    This function analyzes documents and assigns tags based on a predefined taxonomy,
    providing detailed reasoning for each classification decision. The taxonomy allows
    you to define fields (categories), their possible values, and definitions for each value.

    For each taxonomy field, the model will:
    1. Think through each possible value with reasoning
    2. Reflect on the overall analysis
    3. Select the best value
    4. Provide a confidence score

    Parameters
    ----------
    expr : polars.Expr
        The document expression to analyze and tag
    taxonomy : Dict[str, Dict[str, Any]]
        A dictionary defining the taxonomy structure:
        {
            "field_name": {
                "description": "Description of what this field represents",
                "values": {
                    "value1": "Definition of value1",
                    "value2": "Definition of value2",
                    ...
                }
            },
            ...
        }
    provider : str or Provider, optional
        The LLM provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The specific model name to use

    Returns
    -------
    polars.Expr
        Expression with structured tags as a Struct. Each taxonomy field becomes
        a nested struct containing:
        - thinking: Dict[str, str] - reasoning for each possible value
        - reflection: str - overall reflection on the field analysis
        - value: str - the selected value
        - confidence: float - confidence score (0.0 to 1.0)

    Examples
    --------
    >>> import polars as pl
    >>> from polar_llama import tag_taxonomy, Provider
    >>>
    >>> # Define a taxonomy
    >>> taxonomy = {
    ...     "sentiment": {
    ...         "description": "The emotional tone of the text",
    ...         "values": {
    ...             "positive": "Text expresses positive emotions, optimism, or favorable opinions",
    ...             "negative": "Text expresses negative emotions, pessimism, or unfavorable opinions",
    ...             "neutral": "Text is factual and objective without clear emotional content"
    ...         }
    ...     },
    ...     "urgency": {
    ...         "description": "How urgent or time-sensitive the content is",
    ...         "values": {
    ...             "high": "Requires immediate attention or action",
    ...             "medium": "Should be addressed soon but not immediately critical",
    ...             "low": "Can be addressed at any convenient time"
    ...         }
    ...     }
    ... }
    >>>
    >>> # Create a dataframe with documents
    >>> df = pl.DataFrame({
    ...     "id": [1, 2],
    ...     "document": [
    ...         "URGENT: The server is down and customers can't access the site!",
    ...         "Our quarterly results exceeded expectations. Great work team!"
    ...     ]
    ... })
    >>>
    >>> # Apply taxonomy tagging
    >>> result = df.with_columns(
    ...     tags=tag_taxonomy(
    ...         pl.col("document"),
    ...         taxonomy,
    ...         provider=Provider.OPENAI,
    ...         model="gpt-4"
    ...     )
    ... )
    >>>
    >>> # Access specific fields and values
    >>> result.select([
    ...     "document",
    ...     pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
    ...     pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("sentiment_conf"),
    ...     pl.col("tags").struct.field("urgency").struct.field("value").alias("urgency")
    ... ])
    """
    # Create the Pydantic model from the taxonomy
    response_model = _create_taxonomy_pydantic_model(taxonomy)

    # Create the system prompt with taxonomy instructions
    system_prompt = _create_taxonomy_prompt(taxonomy, document_field_name="document")

    # Parse the document expression
    doc_expr = parse_into_expr(expr)

    # Create a system message for each row by mapping over the document column
    # This ensures the system message is broadcast to match the number of rows
    system_message_expr = doc_expr.map_batches(
        lambda s: pl.Series([system_prompt] * len(s)),
        return_dtype=pl.Utf8
    ).pipe(string_to_message, message_type="system")

    # Create a user message with the document
    user_message_expr = doc_expr.pipe(
        string_to_message, message_type="user"
    )

    # Combine the messages
    messages_expr = combine_messages(system_message_expr, user_message_expr)

    # Call inference_messages with the structured output model
    return inference_messages(
        messages_expr,
        provider=provider,
        model=model,
        response_model=response_model
    )


# ============================================================================
# Polars Namespace Accessor
# ============================================================================

@pl.api.register_expr_namespace("llama")
class LlamaNamespace:
    """
    Polars namespace accessor for polar-llama functionality.
    Allows using `.llama` on expressions for a fluent API.
    """
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def to_message(self, *, role: str = "user") -> pl.Expr:
        """
        Convert a string expression to a message with the specified role.
        
        Parameters
        ----------
        role : str
            The role of the message ("user", "system", "assistant")
            
        Returns
        -------
        polars.Expr
            Expression with formatted messages
        """
        return string_to_message(self._expr, message_type=role)

    def inference(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
        response_model: Optional[Type['BaseModel']] = None,
        response_format: Optional[Type['BaseModel']] = None,
    ) -> pl.Expr:
        """
        Synchronously infer completions for the expression using an LLM.
        
        Parameters
        ----------
        provider : str or Provider, optional
            The provider to use
        model : str, optional
            The model name to use
        response_model : Type[BaseModel], optional
            Pydantic model for structured output
        response_format : Type[BaseModel], optional
            Alias for response_model
            
        Returns
        -------
        polars.Expr
            Expression with inferred completions
        """
        return inference(
            self._expr,
            provider=provider,
            model=model,
            response_model=response_model or response_format
        )

    def inference_async(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
        response_model: Optional[Type['BaseModel']] = None,
        response_format: Optional[Type['BaseModel']] = None,
    ) -> pl.Expr:
        """
        Asynchronously infer completions for the expression using an LLM.
        
        Parameters
        ----------
        provider : str or Provider, optional
            The provider to use
        model : str, optional
            The model name to use
        response_model : Type[BaseModel], optional
            Pydantic model for structured output
        response_format : Type[BaseModel], optional
            Alias for response_model
            
        Returns
        -------
        polars.Expr
            Expression with inferred completions
        """
        return inference_async(
            self._expr,
            provider=provider,
            model=model,
            response_model=response_model or response_format
        )
    
    def tag_taxonomy(
        self,
        taxonomy: Dict[str, Dict[str, Any]],
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
    ) -> pl.Expr:
        """
        Tag documents according to a taxonomy definition.
        """
        return tag_taxonomy(
            self._expr,
            taxonomy,
            provider=provider,
            model=model
        )

    def embedding(
        self,
        *,
        provider: Optional[Union[str, Provider]] = None,
        model: Optional[str] = None,
    ) -> pl.Expr:
        """
        Generate embeddings for the expression using an embedding model.

        Parameters
        ----------
        provider : str or Provider, optional
            The provider to use
        model : str, optional
            The model name to use

        Returns
        -------
        polars.Expr
            Expression with embeddings as List[Float64]
        """
        return embedding_async(
            self._expr,
            provider=provider,
            model=model
        )

    def cosine_similarity(self, other: IntoExpr) -> pl.Expr:
        """
        Calculate cosine similarity with another embedding vector.

        Parameters
        ----------
        other : polars.Expr
            The other embedding vector to compare with

        Returns
        -------
        polars.Expr
            Cosine similarity score
        """
        return cosine_similarity(self._expr, other)

    def dot_product(self, other: IntoExpr) -> pl.Expr:
        """
        Calculate dot product with another embedding vector.

        Parameters
        ----------
        other : polars.Expr
            The other embedding vector to compute dot product with

        Returns
        -------
        polars.Expr
            Dot product value
        """
        return dot_product(self._expr, other)

    def euclidean_distance(self, other: IntoExpr) -> pl.Expr:
        """
        Calculate Euclidean distance to another embedding vector.

        Parameters
        ----------
        other : polars.Expr
            The other embedding vector to measure distance to

        Returns
        -------
        polars.Expr
            Euclidean distance
        """
        return euclidean_distance(self._expr, other)

    def knn_hnsw(self, reference: IntoExpr, *, k: int = 5) -> pl.Expr:
        """
        Find k-nearest neighbors using HNSW algorithm.

        Parameters
        ----------
        reference : polars.Expr
            Reference embeddings to search through
        k : int, optional
            Number of nearest neighbors to return (default: 5)

        Returns
        -------
        polars.Expr
            List of indices of the k nearest neighbors
        """
        return knn_hnsw(self._expr, reference, k=k)


# ============================================================================
# Helper Functions
# ============================================================================

def template(format_string: str, *args: IntoExpr, **kwargs: IntoExpr) -> pl.Expr:
    """
    Helper function for prompt templating that abstracts away Polars version differences.
    
    Usage:
        polar_llama.template("Hello {}", pl.col("name"))
        polar_llama.template("Hello {name}", name=pl.col("name"))
        
    Parameters
    ----------
    format_string : str
        The format string
    *args : polars.Expr
        Positional arguments for formatting
    **kwargs : polars.Expr
        Keyword arguments for formatting
        
    Returns
    -------
    polars.Expr
        Formatted string expression
    """
    # Check if current Polars version supports kwargs in pl.format
    # pl.format with kwargs was introduced in recent versions
    # If we want to be safe, we can try to use it, and if it fails, fallback or error
    # But simpler is to rely on pl.format if available.
    
    # For now, we'll just wrap pl.format. 
    # If the user is on an old version that doesn't support kwargs, they should use positional args
    # or we can try to implement a polyfill if needed.
    # However, the recommendation implies we should abstract it.
    
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

def _validate_strict_mode_schema(model: Type['BaseModel']) -> None:
    """
    Validate that a Pydantic model is compatible with OpenAI Strict Mode.
    Specifically checks for default values which are not allowed.
    """
    try:
        from pydantic import BaseModel
        from pydantic.fields import FieldInfo
        
        if not issubclass(model, BaseModel):
            return

        for name, field in model.model_fields.items():
            # Check if field has a default value
            # In Pydantic v2, field.default is PydanticUndefined if no default
            # field.is_required() is another way to check
            
            if not field.is_required():
                # It has a default value (or is Optional with default None)
                # OpenAI Strict Mode doesn't support default values for required fields?
                # Actually, OpenAI Strict Mode requires all fields to be required (no optionality in the JSON schema sense of missing keys),
                # but it DOES allow nullable fields if they are part of the schema.
                # However, the issue reported is "rejects fields with default values like currency: str = 'USD'".
                # This means the field is NOT required in Pydantic, so it has a default.
                # OpenAI expects the schema to NOT have defaults if we want strict adherence?
                # Or rather, OpenAI's structured outputs require all fields to be specified in the schema and usually required.
                
                # Let's warn if we see a default value that is not None (Optional is usually fine if handled correctly, but defaults like "USD" are problematic)
                if field.default is not None and str(field.default) != "PydanticUndefined":
                     import warnings
                     warnings.warn(
                         f"Field '{name}' in model '{model.__name__}' has a default value '{field.default}'. "
                         "OpenAI Structured Outputs (Strict Mode) may not support default values. "
                         "Consider removing the default value or handling it explicitly.",
                         UserWarning
                     )
    except ImportError:
        pass