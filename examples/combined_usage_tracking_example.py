"""
Example demonstrating token usage tracking with Pydantic structured outputs.

This shows how to use both track_usage=True and response_model together
to get validated structured outputs AND token consumption metrics.
"""

import polars as pl
import polar_llama as pll
from pydantic import BaseModel

# Define a structured output schema
class MovieRecommendation(BaseModel):
    title: str
    genre: str
    year: int
    reason: str

# Create a sample dataframe
df = pl.DataFrame({
    "prompt": [
        "Recommend a great sci-fi movie from the 2010s",
        "Recommend a classic comedy film",
    ]
})

print("Example: Structured Output + Usage Tracking")
print("=" * 60)

# Use BOTH response_model and track_usage together
result = df.with_columns(
    pll.inference_async(
        pl.col("prompt"),
        provider="openai",
        model="gpt-4o-mini",
        response_model=MovieRecommendation,  # Get structured output
        track_usage=True  # AND track token usage
    ).alias("response")
)

print(result)
print()

# The response is now a struct with:
# - content: a Struct with the MovieRecommendation fields (title, genre, year, reason)
# - prompt_tokens, completion_tokens, total_tokens

print("Extracting Structured Fields and Usage:")
print("=" * 60)

# Extract the nested struct fields
expanded = result.with_columns([
    # Extract the structured content fields
    pl.col("response").struct.field("content").struct.field("title").alias("title"),
    pl.col("response").struct.field("content").struct.field("genre").alias("genre"),
    pl.col("response").struct.field("content").struct.field("year").alias("year"),
    pl.col("response").struct.field("content").struct.field("reason").alias("reason"),
    # Extract the usage fields
    pl.col("response").struct.field("prompt_tokens").alias("prompt_tokens"),
    pl.col("response").struct.field("completion_tokens").alias("completion_tokens"),
    pl.col("response").struct.field("total_tokens").alias("total_tokens"),
])

print(expanded.select(["title", "genre", "year", "prompt_tokens", "total_tokens"]))
print()

# Calculate costs
INPUT_PRICE_PER_1M = 0.150
OUTPUT_PRICE_PER_1M = 0.600

cost_df = expanded.with_columns(
    cost_usd=(
        (pl.col("prompt_tokens") * INPUT_PRICE_PER_1M / 1_000_000) +
        (pl.col("completion_tokens") * OUTPUT_PRICE_PER_1M / 1_000_000)
    )
)

print("With Cost Calculation:")
print("=" * 60)
print(cost_df.select(["title", "year", "total_tokens", "cost_usd"]))
print()

print(f"Total cost: ${cost_df['cost_usd'].sum():.6f}")
