"""
Label-first taxonomy definition generation.

This module provides utilities to generate taxonomy definitions by first collecting
labels and then asking the model to explain why each label fits, finally synthesizing
definitions from these explanations.
"""

from typing import Dict, List, Optional, Union, Any
import polars as pl

if hasattr(pl, 'IntoExpr'):
    from polars import IntoExpr
else:
    IntoExpr = Any


def explain_label_choice(
    content_col: IntoExpr,
    label_col: IntoExpr,
    field_name: str,
    all_values: Union[List[str], Dict[str, str]],
    *,
    provider: Optional[Union[str, Any]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    For each row, ask the model why the content is labeled with the given value
    and not any of the other possible values.

    This generates row-level explanations that can later be aggregated to create
    taxonomy definitions.

    Parameters
    ----------
    content_col : polars.Expr
        The content/text to analyze
    label_col : polars.Expr
        The actual label assigned to this content
    field_name : str
        The name of the taxonomy field (e.g., "sentiment", "category")
    all_values : List[str] or Dict[str, str]
        All possible values for this field. If dict, keys are value names.
    provider : str or Provider, optional
        The LLM provider to use
    model : str, optional
        The specific model name

    Returns
    -------
    polars.Expr
        Expression containing explanation for why this label was chosen

    Examples
    --------
    >>> import polars as pl
    >>> import polar_llama as pl_llama
    >>>
    >>> df = pl.DataFrame({
    ...     "text": ["Great product!", "Terrible experience"],
    ...     "sentiment": ["positive", "negative"]
    ... })
    >>>
    >>> result = df.with_columns(
    ...     explanation=pl_llama.explain_label_choice(
    ...         pl.col("text"),
    ...         pl.col("sentiment"),
    ...         field_name="sentiment",
    ...         all_values=["positive", "negative", "neutral"],
    ...         provider="anthropic",
    ...         model="claude-haiku-4-5-20251001"
    ...     )
    ... )
    """
    # Convert all_values to list of value names
    if isinstance(all_values, dict):
        value_names = list(all_values.keys())
    else:
        value_names = list(all_values)

    # Build the prompt template
    prompt_template = f"""You are analyzing why content is labeled with a specific value.

Field: {field_name}
Possible values: {", ".join(value_names)}

Content: {{content}}
Assigned label: {{label}}

Explain why this content should be labeled as "{{label}}" and not any of the other possible values ({", ".join([v for v in value_names])}).

Focus on the specific characteristics of the content that justify this label over the alternatives. Be specific and cite examples from the content.

Your explanation:"""

    # Import at usage time to avoid circular imports
    from polar_llama import inference, parse_into_expr

    # Create prompt column using content and label
    content_expr = parse_into_expr(content_col)
    label_expr = parse_into_expr(label_col)

    # Build prompt manually using string concatenation (compatible with all Polars versions)
    prompt_expr = (
        pl.lit(f"""You are analyzing why content is labeled with a specific value.

Field: {field_name}
Possible values: {", ".join(value_names)}

Content: """) +
        content_expr +
        pl.lit("\nAssigned label: ") +
        label_expr +
        pl.lit(f"""

Explain why this content should be labeled as \"""") +
        label_expr +
        pl.lit(f"""\" and not any of the other possible values ({", ".join([v for v in value_names])}).

Focus on the specific characteristics of the content that justify this label over the alternatives. Be specific and cite examples from the content.

Your explanation:""")
    )

    # Call inference to get explanation
    return inference(
        prompt_expr,
        provider=provider,
        model=model
    )


def generate_taxonomy_from_labels(
    df: pl.DataFrame,
    field_name: str,
    label_col: str,
    explanation_col: str,
    field_description: Optional[str] = None,
    *,
    provider: Optional[Union[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate a taxonomy definition from a DataFrame of labels and their explanations.

    This aggregates all explanations per label and asks the LLM to synthesize
    a clear, concise definition for each value suitable for taxonomy-based tagging.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing labels and explanations
    field_name : str
        The name of the taxonomy field
    label_col : str
        Column name containing the labels
    explanation_col : str
        Column name containing the explanations from explain_label_choice()
    field_description : str, optional
        Description of what this field represents
    provider : str or Provider, optional
        The LLM provider to use
    model : str, optional
        The specific model name

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A taxonomy dictionary suitable for use with tag_taxonomy()

    Examples
    --------
    >>> import polars as pl
    >>> import polar_llama as pl_llama
    >>>
    >>> # First, get explanations
    >>> df = pl.DataFrame({
    ...     "text": ["Great!", "Awful", "It's okay"],
    ...     "sentiment": ["positive", "negative", "neutral"]
    ... })
    >>>
    >>> df = df.with_columns(
    ...     explanation=pl_llama.explain_label_choice(
    ...         pl.col("text"),
    ...         pl.col("sentiment"),
    ...         field_name="sentiment",
    ...         all_values=["positive", "negative", "neutral"],
    ...         provider="anthropic",
    ...         model="claude-haiku-4-5-20251001"
    ...     )
    ... )
    >>>
    >>> # Then generate taxonomy
    >>> taxonomy = pl_llama.generate_taxonomy_from_labels(
    ...     df,
    ...     field_name="sentiment",
    ...     label_col="sentiment",
    ...     explanation_col="explanation",
    ...     field_description="The emotional tone of the text",
    ...     provider="anthropic",
    ...     model="claude-sonnet-4-5-20250929"
    ... )
    """
    # Import at usage time to avoid circular imports
    from polar_llama import inference

    # Group by label and collect all explanations
    grouped = df.group_by(label_col).agg(
        pl.col(explanation_col).alias("explanations")
    )

    # Get all unique labels
    labels = grouped[label_col].to_list()

    # Build taxonomy structure
    taxonomy_values = {}

    for row in grouped.iter_rows(named=True):
        label = row[label_col]
        explanations = row["explanations"]

        # Build prompt to synthesize definition
        prompt = f"""You are creating a concise, clear definition for a taxonomy value based on multiple examples and explanations.

Taxonomy field: {field_name}
{f"Field description: {field_description}" if field_description else ""}
Value: {label}

Below are explanations from multiple examples of why content was labeled as "{label}":

{chr(10).join(f"- {exp}" for exp in explanations)}

Based on these explanations, create a single, concise definition (1-2 sentences) that captures what makes content belong to the "{label}" category. This definition will be used to classify new content, so it should be clear and distinguishing.

Focus on the essential characteristics that differentiate this value from others in the taxonomy.

Definition:"""

        # Use inference to generate definition
        temp_df = pl.DataFrame({"prompt": [prompt]})
        result_df = temp_df.with_columns(
            inference(
                pl.col("prompt"),
                provider=provider,
                model=model
            ).alias("definition")
        )

        definition = result_df["definition"][0].strip()
        taxonomy_values[label] = definition

    # Build final taxonomy structure
    taxonomy = {
        field_name: {
            "description": field_description or f"Classification field: {field_name}",
            "values": taxonomy_values
        }
    }

    return taxonomy
