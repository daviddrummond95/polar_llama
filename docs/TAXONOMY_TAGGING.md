# Taxonomy-based Tagging

## Overview

Taxonomy-based tagging is a powerful feature in polar_llama that allows you to classify documents according to a custom taxonomy with detailed reasoning, reflection, and confidence scores. This feature is particularly useful for:

- **Content classification** - Categorize articles, posts, or documents
- **Customer support** - Route and prioritize support tickets
- **Email triage** - Automatically route and prioritize emails
- **Sentiment analysis** - Analyze tone and emotional content with reasoning
- **Multi-label classification** - Apply multiple categorization schemes simultaneously

## Key Features

✨ **Detailed Reasoning**: For each possible value in each field, the model provides its reasoning

🔍 **Reflection**: After considering all options, the model reflects on its analysis

📊 **Confidence Scores**: Each classification includes a confidence score (0.0 to 1.0)

⚡ **Parallel Processing**: Multiple documents and fields are processed in parallel automatically

🎯 **Type Safety**: Returns structured Polars Struct columns for easy manipulation

## Quick Start

### Basic Example

```python
import polars as pl
from polar_llama import tag_taxonomy, Provider

# Define your taxonomy
taxonomy = {
    "sentiment": {
        "description": "The emotional tone of the text",
        "values": {
            "positive": "Text expresses positive emotions or favorable opinions",
            "negative": "Text expresses negative emotions or unfavorable opinions",
            "neutral": "Text is factual and objective without clear emotional content"
        }
    },
    "urgency": {
        "description": "How urgent the content is",
        "values": {
            "high": "Requires immediate attention",
            "medium": "Should be addressed soon",
            "low": "Can be addressed at any time"
        }
    }
}

# Create a dataframe
df = pl.DataFrame({
    "id": [1, 2],
    "message": [
        "URGENT: Server is down!",
        "Thanks for your help yesterday."
    ]
})

# Apply taxonomy tagging
result = df.with_columns(
    tags=tag_taxonomy(
        pl.col("message"),
        taxonomy,
        provider=Provider.GROQ,
        model="openai/gpt-oss-120b"
    )
)

# Extract specific values
result.select([
    "message",
    pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
    pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("confidence"),
    pl.col("tags").struct.field("urgency").struct.field("value").alias("urgency")
])
```

## Taxonomy Definition

A taxonomy is defined as a dictionary with the following structure:

```python
taxonomy = {
    "field_name": {
        "description": "What this field represents",
        "values": {
            "value1": "Definition of value1",
            "value2": "Definition of value2",
            # ... more values
        }
    },
    # ... more fields
}
```

### Components

- **field_name**: The name of the classification dimension (e.g., "sentiment", "category", "priority")
- **description**: A clear explanation of what this field represents
- **values**: A dictionary mapping value names to their definitions

### Design Tips

1. **Clear Definitions**: Make value definitions specific and mutually exclusive
2. **Appropriate Granularity**: 3-5 values per field works well; too many can confuse the model
3. **Balanced Options**: Try to provide balanced options that cover the full range
4. **Domain-Specific**: Tailor definitions to your specific use case

## Output Structure

Each tagged document returns a Struct with the following nested structure:

```python
{
    "field_name": {
        "thinking": {
            "value1": "Reasoning about why value1 might apply...",
            "value2": "Reasoning about why value2 might apply...",
            # ... reasoning for each possible value
        },
        "reflection": "Overall reflection on the analysis of this field...",
        "value": "selected_value",  # The chosen value
        "confidence": 0.87  # Confidence score (0.0 to 1.0)
    },
    # ... more fields
}
```

### Field Descriptions

- **thinking**: A dictionary with reasoning for each possible value in the taxonomy
- **reflection**: The model's overall reflection after considering all options
- **value**: The selected value (one of the values from the taxonomy)
- **confidence**: How confident the model is in its selection (0.0 = not confident, 1.0 = very confident)

## Accessing Results

### Extract Specific Fields

```python
# Get just the selected value
sentiment = result_df.select(
    pl.col("tags").struct.field("sentiment").struct.field("value")
)

# Get value and confidence together
sentiment_analysis = result_df.select([
    pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
    pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("confidence")
])
```

### Access Detailed Reasoning

```python
# Get the thinking for a specific field
thinking = result_df.select(
    pl.col("tags").struct.field("sentiment").struct.field("thinking")
)

# Get the reflection
reflection = result_df.select(
    pl.col("tags").struct.field("sentiment").struct.field("reflection")
)
```

### Multiple Fields at Once

```python
# Create a clean summary view
summary = result_df.select([
    "id",
    "document",
    pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
    pl.col("tags").struct.field("urgency").struct.field("value").alias("urgency"),
    pl.col("tags").struct.field("category").struct.field("value").alias("category")
])
```

## Advanced Usage

### Filtering by Confidence

Filter out classifications where the model is uncertain:

```python
# Only keep high-confidence results
high_confidence = result_df.filter(
    pl.col("tags").struct.field("sentiment").struct.field("confidence") > 0.8
)
```

### Combining Multiple Conditions

```python
# Find negative, urgent items with high confidence
critical = result_df.filter(
    (pl.col("tags").struct.field("sentiment").struct.field("value") == "negative") &
    (pl.col("tags").struct.field("urgency").struct.field("value") == "high") &
    (pl.col("tags").struct.field("urgency").struct.field("confidence") > 0.7)
)
```

### Aggregating by Category

```python
# Count documents by sentiment
sentiment_counts = result_df.groupby(
    pl.col("tags").struct.field("sentiment").struct.field("value")
).count()

# Average confidence by category
avg_confidence = result_df.groupby(
    pl.col("tags").struct.field("category").struct.field("value")
).agg(
    pl.col("tags").struct.field("category").struct.field("confidence").mean()
)
```

## Performance

### Parallel Processing

The taxonomy tagging system automatically processes:

- **Rows in parallel**: Multiple documents are processed concurrently
- **Fields efficiently**: All taxonomy fields are processed in a single API call per document

### Optimization Tips

1. **Batch Processing**: Process multiple documents at once for maximum parallelism
2. **Field Design**: Keep the number of values per field reasonable (3-7 is optimal)
3. **Provider Selection**: Different providers have different performance characteristics
4. **Model Selection**: Larger models may provide better reasoning but are slower

## Use Cases

### 1. Customer Support Routing

```python
taxonomy = {
    "department": {
        "description": "Which department should handle this",
        "values": {
            "sales": "Product inquiries and purchases",
            "support": "Technical issues and bugs",
            "billing": "Payment and account questions"
        }
    },
    "priority": {
        "description": "How urgent this is",
        "values": {
            "urgent": "Service down or critical issue",
            "high": "Significant problem affecting work",
            "normal": "Standard request or question"
        }
    }
}
```

### 2. Content Classification

```python
taxonomy = {
    "category": {
        "description": "Main topic area",
        "values": {
            "technology": "Tech, software, or digital topics",
            "business": "Business, finance, or economics",
            "lifestyle": "Health, wellness, or personal topics"
        }
    },
    "content_type": {
        "description": "Format and purpose",
        "values": {
            "tutorial": "Step-by-step instructional content",
            "analysis": "In-depth examination of a topic",
            "news": "Timely reporting of events"
        }
    }
}
```

### 3. Social Media Analysis

```python
taxonomy = {
    "sentiment": {
        "description": "Emotional tone",
        "values": {
            "positive": "Positive emotions or opinions",
            "negative": "Negative emotions or criticism",
            "neutral": "Factual without clear emotion"
        }
    },
    "topic": {
        "description": "Main subject discussed",
        "values": {
            "product": "Discussion of product features",
            "service": "Customer service experience",
            "brand": "General brand perception"
        }
    },
    "intent": {
        "description": "What the author wants",
        "values": {
            "complaint": "Expressing dissatisfaction",
            "praise": "Sharing positive experience",
            "question": "Seeking information"
        }
    }
}
```

## Error Handling

Like other structured outputs in polar_llama, taxonomy tagging includes error fields:

```python
# Check for errors
errors = result_df.filter(
    pl.col("tags").struct.field("_error").is_not_null()
)

# Get error details
error_details = errors.select([
    "id",
    pl.col("tags").struct.field("_error").alias("error"),
    pl.col("tags").struct.field("_details").alias("details")
])
```

## API Reference

### `tag_taxonomy()`

```python
def tag_taxonomy(
    expr: IntoExpr,
    taxonomy: Dict[str, Dict[str, Any]],
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
) -> pl.Expr
```

**Parameters:**

- `expr`: The document expression to analyze and tag
- `taxonomy`: Dictionary defining the taxonomy structure (see Taxonomy Definition)
- `provider`: The LLM provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
- `model`: The specific model name to use

**Returns:**

- Polars Expression with structured tags as a Struct column

## Examples

See the `examples/taxonomy_tagging_example.py` file for complete, runnable examples including:

1. Basic taxonomy tagging
2. Content classification
3. Email routing and prioritization
4. Advanced filtering with confidence scores

Run the examples with:

```bash
cd examples
python taxonomy_tagging_example.py
```

## Best Practices

1. **Start Simple**: Begin with 2-3 fields and expand as needed
2. **Test Definitions**: Verify that your value definitions are clear and distinguishable
3. **Use Confidence Scores**: Filter or flag low-confidence results for review
4. **Validate Results**: Spot-check classifications to ensure quality
5. **Iterate**: Refine your taxonomy based on results
6. **Handle Errors**: Always check for and handle error cases

## Troubleshooting

### Low Confidence Scores

- **Cause**: Value definitions may be too similar or ambiguous
- **Solution**: Make definitions more distinct and specific

### Incorrect Classifications

- **Cause**: Value definitions don't match your intent
- **Solution**: Revise definitions to be more explicit about what qualifies

### Slow Performance

- **Cause**: Too many fields or values
- **Solution**: Reduce taxonomy complexity or use a faster model

### API Errors

- **Cause**: Model doesn't support structured outputs
- **Solution**: Use a provider/model that supports structured outputs (Groq, OpenAI, Anthropic)

## Contributing

If you have ideas for improving taxonomy-based tagging, please open an issue or pull request on GitHub!
