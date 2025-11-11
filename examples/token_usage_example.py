"""
Example demonstrating token usage tracking with polar_llama.

This example shows how to track token usage (prompt_tokens, completion_tokens, total_tokens)
when making inference requests to LLMs.
"""

import polars as pl
import polar_llama as pll

# Create a sample dataframe with prompts
df = pl.DataFrame({
    "prompt": [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What's 2+2?",
    ]
})

print("Example 1: Basic Usage Tracking")
print("=" * 50)

# Run inference with usage tracking enabled
result_with_usage = df.with_columns(
    pll.inference_async(
        pl.col("prompt"),
        provider="openai",
        model="gpt-4o-mini",
        track_usage=True  # Enable usage tracking
    ).alias("response")
)

print(result_with_usage)
print()

# Access the usage fields
print("Example 2: Extracting Usage Statistics")
print("=" * 50)

usage_df = result_with_usage.with_columns([
    pl.col("response").struct.field("content").alias("content"),
    pl.col("response").struct.field("prompt_tokens").alias("prompt_tokens"),
    pl.col("response").struct.field("completion_tokens").alias("completion_tokens"),
    pl.col("response").struct.field("total_tokens").alias("total_tokens"),
])

print(usage_df)
print()

# Calculate total cost (example pricing for gpt-4o-mini)
print("Example 3: Cost Calculation")
print("=" * 50)

# GPT-4o-mini pricing (as of example date):
# $0.150 per 1M input tokens
# $0.600 per 1M output tokens
INPUT_PRICE_PER_1M = 0.150
OUTPUT_PRICE_PER_1M = 0.600

cost_df = usage_df.with_columns([
    ((pl.col("prompt_tokens") * INPUT_PRICE_PER_1M / 1_000_000) +
     (pl.col("completion_tokens") * OUTPUT_PRICE_PER_1M / 1_000_000)).alias("cost_usd")
])

print(cost_df.select(["content", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"]))
print()

# Calculate total usage
print("Example 4: Aggregate Statistics")
print("=" * 50)

totals = usage_df.select([
    pl.col("prompt_tokens").sum().alias("total_prompt_tokens"),
    pl.col("completion_tokens").sum().alias("total_completion_tokens"),
    pl.col("total_tokens").sum().alias("total_tokens"),
])

print(totals)
print()

print("Example 5: Without Usage Tracking (Default)")
print("=" * 50)

# Run inference without usage tracking (default behavior)
result_without_usage = df.with_columns(
    pll.inference_async(
        pl.col("prompt"),
        provider="openai",
        model="gpt-4o-mini",
        track_usage=False  # Disable usage tracking (default)
    ).alias("response")
)

print(result_without_usage)
print("Note: Response is just a string, not a struct with usage fields")
