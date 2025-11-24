"""
Hierarchical taxonomy tagging.

This module provides utilities to work with hierarchical (nested) taxonomies,
including suggesting groupings for flat taxonomies and tagging content at
multiple levels of hierarchy.
"""

from typing import Dict, List, Optional, Union, Any
import json
import polars as pl

if hasattr(pl, 'IntoExpr'):
    from polars import IntoExpr
else:
    IntoExpr = Any


def suggest_taxonomy_hierarchy(
    flat_taxonomy: Dict[str, Dict[str, Any]],
    field_name: str,
    *,
    num_groups: Optional[int] = None,
    provider: Optional[Union[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Suggest hierarchical groupings for a flat taxonomy.

    Takes a flat taxonomy with many values and asks the LLM to suggest
    logical groupings, creating a hierarchical taxonomy structure.

    Parameters
    ----------
    flat_taxonomy : Dict[str, Dict[str, Any]]
        A flat taxonomy with a single field containing many values
    field_name : str
        The name of the field in the flat taxonomy to hierarchize
    num_groups : int, optional
        Suggested number of top-level groups (LLM may adjust)
    provider : str or Provider, optional
        The LLM provider to use
    model : str, optional
        The specific model name

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A hierarchical taxonomy with nested structure

    Examples
    --------
    >>> import polar_llama as pl_llama
    >>>
    >>> flat_taxonomy = {
    ...     "medical_specialty": {
    ...         "description": "Medical specialty area",
    ...         "values": {
    ...             "Cardiology": "Heart and cardiovascular system",
    ...             "Neurology": "Brain and nervous system",
    ...             "Orthopedics": "Bones and joints",
    ...             "Dermatology": "Skin conditions",
    ...             # ... many more
    ...         }
    ...     }
    ... }
    >>>
    >>> hierarchical = pl_llama.suggest_taxonomy_hierarchy(
    ...     flat_taxonomy,
    ...     field_name="medical_specialty",
    ...     num_groups=3,
    ...     provider="anthropic",
    ...     model="claude-sonnet-4-5-20250929"
    ... )
    """
    # Import at usage time to avoid circular imports
    from polar_llama import inference

    if field_name not in flat_taxonomy:
        raise ValueError(f"Field '{field_name}' not found in taxonomy")

    field_config = flat_taxonomy[field_name]
    values = field_config.get("values", {})
    description = field_config.get("description", "")

    # Build prompt for LLM to suggest groupings
    prompt = f"""You are organizing a flat taxonomy into a hierarchical structure.

Field: {field_name}
Description: {description}
{f"Suggested number of top-level groups: {num_groups}" if num_groups else ""}

Current values ({len(values)} total):
{chr(10).join(f"- **{name}**: {defn}" for name, defn in values.items())}

Create a hierarchical grouping of these values. Suggest logical top-level categories that group related values together.

Return your response as a JSON object with this structure:
{{
  "groups": {{
    "group_name_1": {{
      "description": "Description of this group",
      "values": ["value1", "value2", ...]
    }},
    "group_name_2": {{
      "description": "Description of this group",
      "values": ["value3", "value4", ...]
    }}
  }}
}}

Ensure:
1. Every original value appears in exactly one group
2. Groups are logical and semantically coherent
3. Group names are clear and descriptive
4. Group descriptions explain what unifies the values

JSON response:"""

    # Call LLM to get groupings
    temp_df = pl.DataFrame({"prompt": [prompt]})
    result_df = temp_df.with_columns(
        inference(
            pl.col("prompt"),
            provider=provider,
            model=model
        ).alias("groupings")
    )

    response_text = result_df["groupings"][0]

    # Strip markdown code fences if present
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # Remove ```json
    elif response_text.startswith("```"):
        response_text = response_text[3:]  # Remove ```
    if response_text.endswith("```"):
        response_text = response_text[:-3]  # Remove trailing ```
    response_text = response_text.strip()

    # Parse JSON response
    try:
        grouping_data = json.loads(response_text)
        groups = grouping_data.get("groups", {})
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON. Response: {response_text}")

    # Build hierarchical taxonomy structure
    # Top level: groups
    # Second level: original values within each group
    hierarchical_taxonomy = {
        f"{field_name}_category": {
            "description": f"Top-level category for {field_name}",
            "values": {
                group_name: group_config.get("description", "")
                for group_name, group_config in groups.items()
            }
        }
    }

    # Store the mapping of categories to values for use in hierarchical tagging
    hierarchical_taxonomy["_hierarchy_mapping"] = {
        group_name: {
            "description": group_config.get("description", ""),
            "subcategory_field": field_name,
            "subcategory_values": {
                value_name: values[value_name]
                for value_name in group_config.get("values", [])
                if value_name in values
            }
        }
        for group_name, group_config in groups.items()
    }

    return hierarchical_taxonomy


def tag_hierarchical_taxonomy(
    expr: IntoExpr,
    hierarchical_taxonomy: Dict[str, Any],
    *,
    provider: Optional[Union[str, Any]] = None,
    model: Optional[str] = None,
) -> pl.Expr:
    """
    Tag content using a hierarchical taxonomy.

    First tags with top-level categories, then conditionally tags with
    subcategories based on the top-level result.

    Parameters
    ----------
    expr : polars.Expr
        The content/text to tag
    hierarchical_taxonomy : Dict[str, Any]
        A hierarchical taxonomy structure (from suggest_taxonomy_hierarchy)
    provider : str or Provider, optional
        The LLM provider to use
    model : str, optional
        The specific model name

    Returns
    -------
    polars.Expr
        Expression with hierarchical tags as nested struct

    Examples
    --------
    >>> import polars as pl
    >>> import polar_llama as pl_llama
    >>>
    >>> # Assume we have a hierarchical taxonomy
    >>> hierarchical_taxonomy = {...}  # from suggest_taxonomy_hierarchy
    >>>
    >>> df = pl.DataFrame({
    ...     "text": ["Patient has irregular heartbeat", "Skin rash treatment"]
    ... })
    >>>
    >>> result = df.with_columns(
    ...     tags=pl_llama.tag_hierarchical_taxonomy(
    ...         pl.col("text"),
    ...         hierarchical_taxonomy,
    ...         provider="anthropic",
    ...         model="claude-haiku-4-5-20251001"
    ...     )
    ... )
    """
    # Import at usage time to avoid circular imports
    from polar_llama import tag_taxonomy

    # This is the Python wrapper approach
    # We'll need to do multiple passes:
    # 1. Tag with top-level taxonomy
    # 2. Based on top-level result, tag with appropriate sub-taxonomy

    # Extract top-level taxonomy (everything except _hierarchy_mapping)
    top_level_taxonomy = {
        k: v for k, v in hierarchical_taxonomy.items()
        if k != "_hierarchy_mapping"
    }

    if not top_level_taxonomy:
        raise ValueError("Hierarchical taxonomy must have at least one top-level field")

    # Get the name of the top-level field
    top_level_field = list(top_level_taxonomy.keys())[0]

    # Step 1: Tag with top-level taxonomy
    top_level_result = tag_taxonomy(
        expr,
        top_level_taxonomy,
        provider=provider,
        model=model
    )

    # Now we need to conditionally apply sub-taxonomies based on top-level result
    # This requires accessing the hierarchy mapping
    hierarchy_mapping = hierarchical_taxonomy.get("_hierarchy_mapping", {})

    if not hierarchy_mapping:
        # No subcategories, just return top-level result
        return top_level_result

    # PARALLEL APPROACH: Tag all sub-taxonomies at once using with_columns,
    # then use conditional selection. This guarantees parallel execution!

    # Extract the top-level value for conditional selection
    top_level_value = top_level_result.struct.field(top_level_field).struct.field("value")

    # Build sub-taxonomy expressions for ALL categories
    subcategory_field_name = None
    sub_taxonomy_exprs = {}  # category -> expression

    for category, mapping in hierarchy_mapping.items():
        subcategory_field = mapping.get("subcategory_field")
        subcategory_values = mapping.get("subcategory_values", {})

        if not subcategory_field:
            continue

        if subcategory_field_name is None:
            subcategory_field_name = subcategory_field

        # Create sub-taxonomy for this category
        sub_taxonomy = {
            subcategory_field: {
                "description": mapping.get("description", ""),
                "values": subcategory_values
            }
        }

        # Store the tag_taxonomy expression for this category
        sub_taxonomy_exprs[category] = tag_taxonomy(
            expr,
            sub_taxonomy,
            provider=provider,
            model=model
        )

    if not sub_taxonomy_exprs:
        return top_level_result

    # The key insight: create a list of column expressions to add with with_columns
    # This will execute all sub-taxonomy tagging in PARALLEL
    # Then we use when/then to select the correct result

    # Build when/then chain to select the appropriate sub-result
    # All sub_taxonomy_exprs will be evaluated in parallel when used in a with_columns context
    subcategory_result = None
    for category, sub_expr in sub_taxonomy_exprs.items():
        if subcategory_result is None:
            # First category - start the chain
            subcategory_result = pl.when(top_level_value == category).then(sub_expr)
        else:
            # Subsequent categories - add to chain
            subcategory_result = subcategory_result.when(top_level_value == category).then(sub_expr)

    # Add final otherwise(None) to handle any unmatched cases
    subcategory_result = subcategory_result.otherwise(None)

    # Add subcategory result to combined result
    combined_result = top_level_result.struct.with_fields(
        **{subcategory_field_name: subcategory_result}
    )

    return combined_result
