#!/usr/bin/env python3
"""
Test parallelization with two taxonomy operations.
This is a better test than taxonomy+embedding since both operations take similar time.
"""

import os
from pathlib import Path
import polars as pl
from polar_llama import Provider, tag_taxonomy
import time
import random

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("❌ ANTHROPIC_API_KEY not set")
    exit(1)

# Two different taxonomies
category_taxonomy = {
    "category": {
        "description": "Primary subject area",
        "values": {
            "technology": "Tech, software, computing",
            "health": "Health, wellness, fitness",
            "business": "Business, finance, economics"
        }
    }
}

difficulty_taxonomy = {
    "difficulty": {
        "description": "Content complexity level",
        "values": {
            "beginner": "Easy to understand, basic concepts",
            "intermediate": "Moderate complexity, some expertise needed",
            "advanced": "Complex, requires significant expertise"
        }
    }
}

def generate_unique_docs(n, prefix):
    """Generate unique documents to avoid caching"""
    return [f"{prefix} Document {i}: {random.random()}" for i in range(n)]

print("=" * 80)
print("Two-Taxonomy Parallelization Test")
print("=" * 80)

N = 5  # Small number for faster testing

# Test 1: Category taxonomy only
print("\n📊 Test 1: Category taxonomy only (FRESH DATA)")
docs1 = generate_unique_docs(N, "Cat")
df1 = pl.DataFrame({"content": docs1})
start = time.time()
result1 = df1.with_columns(
    tags=tag_taxonomy(pl.col("content"), category_taxonomy, provider=Provider.ANTHROPIC, model="claude-sonnet-4-5-20250929")
)
time1 = time.time() - start
print(f"✓ {time1:.2f}s")

# Test 2: Difficulty taxonomy only
print("\n📊 Test 2: Difficulty taxonomy only (FRESH DATA)")
docs2 = generate_unique_docs(N, "Diff")
df2 = pl.DataFrame({"content": docs2})
start = time.time()
result2 = df2.with_columns(
    tags=tag_taxonomy(pl.col("content"), difficulty_taxonomy, provider=Provider.ANTHROPIC, model="claude-sonnet-4-5-20250929")
)
time2 = time.time() - start
print(f"✓ {time2:.2f}s")

# Test 3: Both taxonomies with with_columns (should be parallel!)
print("\n📊 Test 3: Both taxonomies with_columns [FRESH DATA] - SHOULD BE PARALLEL")
docs3 = generate_unique_docs(N, "Both")
df3 = pl.DataFrame({"content": docs3})
start = time.time()
result3 = df3.with_columns([
    tag_taxonomy(pl.col("content"), category_taxonomy, provider=Provider.ANTHROPIC, model="claude-sonnet-4-5-20250929").alias("category_tags"),
    tag_taxonomy(pl.col("content"), difficulty_taxonomy, provider=Provider.ANTHROPIC, model="claude-sonnet-4-5-20250929").alias("difficulty_tags")
])
time3 = time.time() - start
print(f"✓ {time3:.2f}s")

# Test 4: Both taxonomies with with_columns_seq (explicitly sequential)
print("\n📊 Test 4: Both taxonomies with_columns_seq [FRESH DATA] - EXPLICITLY SEQUENTIAL")
docs4 = generate_unique_docs(N, "Seq")
df4 = pl.DataFrame({"content": docs4})
start = time.time()
result4 = df4.with_columns_seq([
    tag_taxonomy(pl.col("content"), category_taxonomy, provider=Provider.ANTHROPIC, model="claude-sonnet-4-5-20250929").alias("category_tags"),
    tag_taxonomy(pl.col("content"), difficulty_taxonomy, provider=Provider.ANTHROPIC, model="claude-sonnet-4-5-20250929").alias("difficulty_tags")
])
time4 = time.time() - start
print(f"✓ {time4:.2f}s")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\nTiming Results:")
print(f"  Category taxonomy:       {time1:6.2f}s")
print(f"  Difficulty taxonomy:     {time2:6.2f}s")
print(f"  with_columns():          {time3:6.2f}s  ← Should be parallel")
print(f"  with_columns_seq():      {time4:6.2f}s  ← Explicitly sequential")

expected_parallel = max(time1, time2)
expected_sequential = time1 + time2

print(f"\nExpected Times:")
print(f"  If fully parallel:        {expected_parallel:6.2f}s  (max of both)")
print(f"  If fully sequential:      {expected_sequential:6.2f}s  (sum of both)")

print(f"\nwith_columns() Analysis:")
if time3 < expected_sequential - 1.0:
    speedup = expected_sequential / time3
    efficiency = ((expected_sequential - time3) / (expected_sequential - expected_parallel)) * 100
    print(f"  ✅ PARALLELIZATION DETECTED!")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {efficiency:.1f}% of ideal parallel")
    print(f"  Overhead: {time3 - expected_parallel:.2f}s above ideal")
else:
    print(f"  ❌ No significant parallelization")
    print(f"  Difference from sequential: {abs(time3 - expected_sequential):.2f}s")

print(f"\nwith_columns_seq() Analysis:")
if abs(time4 - expected_sequential) < 2.0:
    print(f"  ✓ Confirmed sequential execution")
else:
    print(f"  ⚠️  Unexpected timing")

print("\n" + "=" * 80)
