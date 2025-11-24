"""
Taxonomy refinement functions for improving category definitions based on prediction errors.

This module provides two key functions:
1. analyze_taxonomy_error - Row-level analysis of what caused a misprediction
2. refine_taxonomy - Aggregate all feedback to generate improved taxonomy definitions
"""

import json
from typing import Dict, Any, List, Optional
import polars as pl
from pydantic import BaseModel, Field


def analyze_taxonomy_error(
    prediction_obj: Dict[str, Any],
    ground_truth: str,
    taxonomy_field: str,
    taxonomy: Dict[str, Any],
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001"
) -> Optional[str]:
    """
    Analyze a single misprediction to understand what was unclear about the definition.

    Args:
        prediction_obj: The prediction struct containing thinking, reflection, value, confidence
        ground_truth: The actual correct value
        taxonomy_field: The field name in the taxonomy (e.g., "sentiment", "medical_specialty")
        taxonomy: The full taxonomy dictionary
        provider: LLM provider to use for analysis
        model: Model to use for analysis

    Returns:
        String feedback about what was unclear in the definition, or None if prediction was correct
    """
    # Import here to avoid circular dependency
    from polar_llama import inference

    # Extract prediction value
    if prediction_obj is None:
        return "Prediction was null - model failed to classify"

    predicted_value = prediction_obj.get('value')
    thinking = prediction_obj.get('thinking', {})
    reflection = prediction_obj.get('reflection', '')
    confidence = prediction_obj.get('confidence', 0.0)

    # If prediction was correct, no analysis needed
    if predicted_value == ground_truth:
        return None

    # Get the ground truth definition
    taxonomy_info = taxonomy.get(taxonomy_field, {})
    category_definitions = taxonomy_info.get('values', {})
    field_description = taxonomy_info.get('description', taxonomy_field)

    ground_truth_definition = category_definitions.get(ground_truth, "No definition available")
    predicted_definition = category_definitions.get(predicted_value, "No definition available") if predicted_value else "N/A"

    # Construct the analysis prompt
    prompt = f"""You are analyzing why a taxonomy classification model made an incorrect prediction.

TASK: {field_description}

GROUND TRUTH CATEGORY: {ground_truth}
Definition: {ground_truth_definition}

PREDICTED CATEGORY: {predicted_value or "None (failed to classify)"}
Definition: {predicted_definition}

MODEL'S REASONING:
{json.dumps(thinking, indent=2) if thinking else "No reasoning provided"}

MODEL'S REFLECTION:
{reflection or "No reflection provided"}

MODEL'S CONFIDENCE: {confidence}

QUESTION: What, if anything, was unclear, ambiguous, or misleading about the "{ground_truth}" category definition that may have caused the model to predict "{predicted_value or 'null'}" instead?

Consider:
1. Is the definition too vague or too specific?
2. Does it overlap conceptually with other categories?
3. Are there missing keywords or context that would help?
4. Is the definition technically accurate but practically confusing?

Provide a brief, actionable analysis (2-3 sentences) focusing on how the definition could be improved."""

    # Use polar_llama.inference to call the LLM
    try:
        # Create a temporary DataFrame with the prompt
        temp_df = pl.DataFrame({"prompt": [prompt]})

        # Call inference using polar_llama
        result_df = temp_df.with_columns(
            inference(
                pl.col("prompt"),
                provider=provider,
                model=model
            ).alias("analysis")
        )

        # Extract the result
        return result_df["analysis"][0]

    except Exception as e:
        return f"Error analyzing: {str(e)[:100]}"


def refine_taxonomy(
    error_feedback: List[Dict[str, Any]],
    original_taxonomy: Dict[str, Any],
    taxonomy_field: str,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5-20250929"
) -> Dict[str, Any]:
    """
    Aggregate all error feedback and use an LLM to generate improved taxonomy definitions.

    Args:
        error_feedback: List of dicts with 'category', 'feedback', 'count' for each misclassified category
        original_taxonomy: The original taxonomy dictionary
        taxonomy_field: The field name in the taxonomy
        provider: LLM provider to use for refinement
        model: Model to use for refinement (use a more capable model here)

    Returns:
        Updated taxonomy dictionary with refined definitions
    """
    # Get original taxonomy info
    taxonomy_info = original_taxonomy.get(taxonomy_field, {})
    original_definitions = taxonomy_info.get('values', {})
    field_description = taxonomy_info.get('description', taxonomy_field)

    # Aggregate feedback by category
    category_feedback = {}
    for item in error_feedback:
        category = item.get('category')
        feedback = item.get('feedback')

        if category and feedback and feedback.strip():
            if category not in category_feedback:
                category_feedback[category] = []
            category_feedback[category].append(feedback)

    # If no feedback, return original
    if not category_feedback:
        return original_taxonomy

    # Construct refinement prompt
    feedback_summary = []
    for category, feedbacks in sorted(category_feedback.items()):
        original_def = original_definitions.get(category, "No definition")
        feedback_summary.append(f"""
Category: {category}
Original Definition: {original_def}
Confusion Points ({len(feedbacks)} mispredictions):
{chr(10).join(f"  - {fb}" for fb in feedbacks)}
""")

    prompt = f"""You are refining taxonomy category definitions based on classification errors.

TAXONOMY FIELD: {taxonomy_field}
FIELD DESCRIPTION: {field_description}

CATEGORIES WITH MISPREDICTIONS:
{"".join(feedback_summary)}

ALL ORIGINAL DEFINITIONS:
{json.dumps(original_definitions, indent=2)}

TASK: Generate improved definitions that address the confusion points while:
1. Maintaining accuracy and clarity
2. Distinguishing clearly from other categories
3. Being concise but comprehensive
4. Using specific, actionable language

Return ONLY a valid JSON object with this exact structure:
{{
  "category_name": "improved definition text",
  "another_category": "improved definition text",
  ...
}}

Include ALL categories from the original taxonomy, not just the ones with errors. For categories without errors, you may keep or slightly improve the original definition.

IMPORTANT: Return ONLY the JSON object, no other text."""

    # Use polar_llama.inference to call the LLM
    try:
        # Import here to avoid circular dependency
        from polar_llama import inference

        # Create a temporary DataFrame with the prompt
        temp_df = pl.DataFrame({"prompt": [prompt]})

        # Call inference using polar_llama
        result_df = temp_df.with_columns(
            inference(
                pl.col("prompt"),
                provider=provider,
                model=model
            ).alias("refinement")
        )

        # Extract the result
        response_text = result_df["refinement"][0]

        # Parse the response
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()

        refined_definitions = json.loads(response_text)

        # Create updated taxonomy
        updated_taxonomy = original_taxonomy.copy()
        updated_taxonomy[taxonomy_field] = {
            'description': field_description,
            'values': refined_definitions
        }

        return updated_taxonomy

    except Exception as e:
        print(f"Error refining taxonomy: {e}")
        return original_taxonomy


def analyze_all_errors(
    df: pl.DataFrame,
    prediction_col: str,
    ground_truth_col: str,
    taxonomy_field: str,
    taxonomy: Dict[str, Any],
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001"
) -> pl.DataFrame:
    """
    Analyze all prediction errors in a DataFrame.

    Args:
        df: DataFrame with predictions and ground truth
        prediction_col: Column name containing prediction objects
        ground_truth_col: Column name containing ground truth values
        taxonomy_field: The field name in the taxonomy
        taxonomy: The taxonomy dictionary
        provider: LLM provider
        model: Model to use

    Returns:
        DataFrame with added 'error_analysis' column containing feedback
    """
    def analyze_row(prediction_obj, ground_truth):
        return analyze_taxonomy_error(
            prediction_obj,
            ground_truth,
            taxonomy_field,
            taxonomy,
            provider,
            model
        )

    return df.with_columns(
        pl.struct([prediction_col, ground_truth_col])
        .map_elements(
            lambda x: analyze_row(x[prediction_col], x[ground_truth_col]),
            return_dtype=pl.Utf8
        )
        .alias("error_analysis")
    )


def aggregate_error_feedback(
    df: pl.DataFrame,
    ground_truth_col: str,
    error_analysis_col: str = "error_analysis"
) -> List[Dict[str, Any]]:
    """
    Aggregate error feedback by category.

    Args:
        df: DataFrame with error analysis
        ground_truth_col: Column with ground truth categories
        error_analysis_col: Column with error feedback

    Returns:
        List of dicts with category, feedback, and count
    """
    # Filter to only rows with feedback (mispredictions)
    errors_df = df.filter(pl.col(error_analysis_col).is_not_null())

    feedback_list = []
    for row in errors_df.iter_rows(named=True):
        category = row[ground_truth_col]
        feedback = row[error_analysis_col]

        if feedback:
            feedback_list.append({
                'category': category,
                'feedback': feedback,
                'count': 1
            })

    return feedback_list
