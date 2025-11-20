#!/usr/bin/env python3
"""
Test to verify that taxonomy tagging and embeddings can execute in parallel.

This script measures:
1. Time for taxonomy tagging alone
2. Time for embeddings alone
3. Time for both together

If parallel: combined_time ≈ max(taxonomy_time, embedding_time)
If sequential: combined_time ≈ taxonomy_time + embedding_time
"""

import os
from pathlib import Path
import polars as pl
from polar_llama import Provider, embedding_async, tag_taxonomy
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("⚠️  dotenv not installed")

# Check for API keys
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set")
    exit(1)
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("❌ ANTHROPIC_API_KEY not set")
    exit(1)

print("=" * 80)
print("Parallel Execution Test: Taxonomy Tagging + Embeddings")
print("=" * 80)

# Create test dataset (10 documents)
test_data = pl.DataFrame({
    "id": list(range(1, 11)),
    "content": [
        "Machine learning is transforming healthcare with AI-powered diagnostics.",
        "The latest smartphone features advanced camera technology and 5G connectivity.",
        "Climate change requires urgent action to reduce carbon emissions globally.",
        "Stock markets showed volatility amid economic uncertainty and inflation concerns.",
        "New fitness app helps users track workouts and maintain healthy habits.",
        "Quantum computing breakthrough promises faster processing for complex problems.",
        "Restaurant chain expands menu with plant-based alternatives and sustainable sourcing.",
        "Education reform focuses on personalized learning and digital classroom tools.",
        "Space exploration mission discovers water ice on distant planetary moon.",
        "Cybersecurity threats evolve as hackers develop sophisticated attack methods."
    ]
})

# Define taxonomy
taxonomy = {
    "category": {
        "description": "Primary subject area",
        "values": {
            "technology": "Tech, software, computing",
            "health": "Health, wellness, fitness",
            "business": "Business, finance, economics",
            "science": "Scientific research and discoveries"
        }
    }
}

print(f"\nTest dataset: {len(test_data)} documents")

# ==============================================================================
# Test 1: Taxonomy Tagging Alone
# ==============================================================================
print("\n" + "-" * 80)
print("Test 1: Taxonomy Tagging Only")
print("-" * 80)

start = time.time()
taxonomy_only = test_data.with_columns(
    tags=tag_taxonomy(
        pl.col("content"),
        taxonomy,
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929"
    )
)
taxonomy_time = time.time() - start

print(f"✓ Completed in {taxonomy_time:.2f}s")
print(f"  Throughput: {len(test_data)/taxonomy_time:.1f} docs/sec")

# ==============================================================================
# Test 2: Embeddings Alone
# ==============================================================================
print("\n" + "-" * 80)
print("Test 2: Embeddings Only")
print("-" * 80)

start = time.time()
embedding_only = test_data.with_columns(
    embedding=embedding_async(
        pl.col("content"),
        provider=Provider.OPENAI,
        model="text-embedding-3-small"
    )
)
embedding_time = time.time() - start

print(f"✓ Completed in {embedding_time:.2f}s")
print(f"  Throughput: {len(test_data)/embedding_time:.1f} docs/sec")

# ==============================================================================
# Test 3: Both Together (Sequential - Eager Execution)
# ==============================================================================
print("\n" + "-" * 80)
print("Test 3: Both Together (Sequential - Eager Execution)")
print("-" * 80)

start = time.time()
# This is sequential: first taxonomy, then embeddings
sequential_result = test_data.with_columns(
    tags=tag_taxonomy(
        pl.col("content"),
        taxonomy,
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929"
    )
).with_columns(
    embedding=embedding_async(
        pl.col("content"),
        provider=Provider.OPENAI,
        model="text-embedding-3-small"
    )
)
sequential_time = time.time() - start

print(f"✓ Completed in {sequential_time:.2f}s")
print(f"  Throughput: {len(test_data)/sequential_time:.1f} docs/sec")

# ==============================================================================
# Test 4: Both Together (Potential Parallel - Lazy Execution)
# ==============================================================================
print("\n" + "-" * 80)
print("Test 4: Both Together (Lazy Execution - Potential Parallel)")
print("-" * 80)

start = time.time()
# Using lazy execution - Polars MAY parallelize these
lazy_result = test_data.lazy().with_columns([
    tag_taxonomy(
        pl.col("content"),
        taxonomy,
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929"
    ).alias("tags"),
    embedding_async(
        pl.col("content"),
        provider=Provider.OPENAI,
        model="text-embedding-3-small"
    ).alias("embedding")
]).collect()
lazy_time = time.time() - start

print(f"✓ Completed in {lazy_time:.2f}s")
print(f"  Throughput: {len(test_data)/lazy_time:.1f} docs/sec")

# ==============================================================================
# Analysis
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\n📊 Timing Results:")
print(f"  Taxonomy only:      {taxonomy_time:6.2f}s")
print(f"  Embeddings only:    {embedding_time:6.2f}s")
print(f"  Both (sequential):  {sequential_time:6.2f}s")
print(f"  Both (lazy):        {lazy_time:6.2f}s")

expected_sequential = taxonomy_time + embedding_time
expected_parallel = max(taxonomy_time, embedding_time)

print(f"\n📈 Expected Times:")
print(f"  If fully sequential: {expected_sequential:6.2f}s  (taxonomy + embeddings)")
print(f"  If fully parallel:   {expected_parallel:6.2f}s  (max of both)")

print(f"\n🔍 Sequential Execution (eager):")
overhead = sequential_time - expected_sequential
overhead_pct = (overhead / expected_sequential) * 100
print(f"  Actual: {sequential_time:.2f}s")
print(f"  Expected: {expected_sequential:.2f}s")
print(f"  Overhead: {overhead:+.2f}s ({overhead_pct:+.1f}%)")
if abs(sequential_time - expected_sequential) < 2.0:
    print(f"  ✓ Confirms sequential execution")
else:
    print(f"  ⚠️  Unexpected timing")

print(f"\n🔍 Lazy Execution:")
if lazy_time < expected_sequential - 2.0:
    speedup = expected_sequential / lazy_time
    efficiency = ((expected_sequential - lazy_time) / (expected_sequential - expected_parallel)) * 100
    print(f"  Actual: {lazy_time:.2f}s")
    print(f"  ✅ PARALLEL EXECUTION DETECTED!")
    print(f"  Speedup: {speedup:.2f}x faster than sequential")
    print(f"  Efficiency: {efficiency:.1f}% of ideal parallel execution")
else:
    print(f"  Actual: {lazy_time:.2f}s")
    print(f"  ❌ NO PARALLEL EXECUTION DETECTED")
    print(f"  Lazy execution is still sequential (similar to eager)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if lazy_time < expected_sequential - 2.0:
    print("✅ Some parallelization detected with lazy execution!")
    print("   Use: df.lazy().with_columns([taxonomy(...), embeddings(...)]).collect()")
    print("\n   However, true column-level parallelism is limited by Polars plugin architecture.")
else:
    print("📊 EXECUTION MODEL:")
    print("   • Operations run SEQUENTIALLY within each document")
    print("   • Each operation PARALLELIZES across all documents (via async/await)")
    print("   • This is due to Polars plugin execution model")
    print("\n   RECOMMENDATION:")
    print("   • Use lazy evaluation for memory efficiency")
    print("   • Both operations are still highly optimized via internal async parallelization")
    print("   • For truly independent parallel work, run operations in separate processes")

print("=" * 80)
