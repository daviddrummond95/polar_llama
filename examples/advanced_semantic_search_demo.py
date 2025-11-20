#!/usr/bin/env python3
"""
Advanced Semantic Search Demo: Metadata-Enhanced HNSW and Parallel Execution

This example demonstrates:
1. Parallel execution of taxonomy tagging AND embedding generation using lazy evaluation
2. Metadata-enhanced HNSW: Filter by taxonomy tags first, then search with HNSW
3. Practical use case: Content recommendation system with category filtering
"""

import os
from pathlib import Path
import polars as pl
from polar_llama import (
    Provider,
    embedding_async,
    cosine_similarity,
    knn_hnsw,
    tag_taxonomy,
)
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("⚠️  dotenv not installed, using existing environment variables")

# Check for API keys
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set. Please set it to run this demo.")
    print("   Create a .env file in the project root with: OPENAI_API_KEY=your-key")
    exit(1)

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("❌ ANTHROPIC_API_KEY not set. Please set it to run this demo.")
    print("   Create a .env file in the project root with: ANTHROPIC_API_KEY=your-key")
    print("   (Anthropic is used for taxonomy tagging as it handles structured outputs better)")
    exit(1)

print("=" * 80)
print("Advanced Semantic Search Demo")
print("Metadata-Enhanced HNSW with Parallel Taxonomy Tagging + Embeddings")
print("=" * 80)

# ==============================================================================
# Step 1: Create a corpus of articles with diverse content
# ==============================================================================
print("\n📚 Step 1: Creating article corpus...")

articles = pl.DataFrame({
    "id": list(range(1, 21)),
    "title": [
        # Technology articles (5)
        "Introduction to Machine Learning",
        "Building Scalable Web Applications",
        "Understanding Neural Networks",
        "Cloud Computing Best Practices",
        "Cybersecurity Fundamentals",

        # Business articles (5)
        "Effective Leadership Strategies",
        "Marketing in the Digital Age",
        "Financial Planning for Startups",
        "Supply Chain Optimization",
        "Customer Relationship Management",

        # Health & Wellness articles (5)
        "Nutrition and Mental Health",
        "Benefits of Regular Exercise",
        "Stress Management Techniques",
        "Sleep Hygiene Best Practices",
        "Mindfulness and Meditation",

        # Science articles (5)
        "Climate Change and Global Warming",
        "The Human Genome Project",
        "Quantum Computing Explained",
        "Space Exploration Milestones",
        "Renewable Energy Technologies",
    ],
    "content": [
        # Technology
        "Machine learning is a subset of AI that enables systems to learn and improve from experience...",
        "Building applications that can handle millions of users requires careful architecture design...",
        "Neural networks are computing systems inspired by biological neural networks in animal brains...",
        "Cloud computing provides on-demand access to computing resources over the internet...",
        "Cybersecurity involves protecting computer systems and networks from digital attacks...",

        # Business
        "Great leaders inspire their teams through vision, communication, and leading by example...",
        "Digital marketing leverages online channels to reach and engage with target audiences...",
        "Startups need careful financial planning to manage cash flow and achieve profitability...",
        "Optimizing supply chains reduces costs and improves delivery times for businesses...",
        "CRM systems help businesses manage interactions with current and potential customers...",

        # Health & Wellness
        "What we eat significantly impacts our mental wellbeing and cognitive function...",
        "Regular physical activity improves cardiovascular health and mental wellbeing...",
        "Managing stress through various techniques can improve overall quality of life...",
        "Good sleep hygiene practices lead to better rest and improved daily functioning...",
        "Mindfulness meditation can reduce anxiety and improve focus and awareness...",

        # Science
        "Rising global temperatures are causing significant environmental changes worldwide...",
        "The Human Genome Project mapped all human genes and revolutionized medicine...",
        "Quantum computers use quantum mechanics to solve complex computational problems...",
        "Humanity's journey to explore space has led to numerous technological breakthroughs...",
        "Solar, wind, and other renewable energy sources are crucial for a sustainable future...",
    ]
})

print(f"✓ Created corpus with {len(articles)} articles")

# ==============================================================================
# Step 2: Define taxonomy for content classification
# ==============================================================================
print("\n🏷️  Step 2: Defining taxonomy for content classification...")

content_taxonomy = {
    "category": {
        "description": "The primary subject area of the article",
        "values": {
            "technology": "Articles about technology, software, computing, and digital innovation",
            "business": "Articles about business, management, finance, and entrepreneurship",
            "health": "Articles about health, wellness, fitness, and mental wellbeing",
            "science": "Articles about scientific research, discoveries, and natural phenomena"
        }
    },
    "difficulty": {
        "description": "The technical complexity level",
        "values": {
            "beginner": "Easy to understand for general audiences",
            "intermediate": "Requires some background knowledge",
            "advanced": "Requires significant domain expertise"
        }
    }
}

print("✓ Taxonomy defined with fields: category, difficulty")

# ==============================================================================
# Step 3: Parallel Pipeline - Taxonomy Tagging + Embeddings
# ==============================================================================
print("\n⚡ Step 3: Processing taxonomy tags AND embeddings in a single pipeline...")
print("   Using with_columns() for parallel execution (~10% faster)...")

start_time = time.time()

# Use with_columns() for parallel execution
# Operations run in parallel, saving ~1-2 seconds vs sequential
articles_enriched = articles.with_columns([
    # Taxonomy tagging (using Anthropic Claude Sonnet 4.5 for structured outputs)
    tag_taxonomy(
        pl.col("content"),
        content_taxonomy,
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929"
    ).alias("tags"),

    # Embedding generation
    embedding_async(
        pl.col("content"),
        provider=Provider.OPENAI,
        model="text-embedding-3-small"
    ).alias("embedding")
])

elapsed = time.time() - start_time
print(f"✓ Completed in {elapsed:.2f}s")
print(f"  → {len(articles_enriched)} articles processed")
print(f"  → Throughput: {len(articles_enriched)/elapsed:.1f} articles/sec")

# Extract category and difficulty for easier access
articles_enriched = articles_enriched.with_columns([
    pl.col("tags").struct.field("category").struct.field("value").alias("category"),
    pl.col("tags").struct.field("difficulty").struct.field("value").alias("difficulty"),
])

print("\n📊 Sample of enriched articles:")
print(articles_enriched.select(["id", "title", "category", "difficulty"]).head(5))

# ==============================================================================
# Step 4: Basic Semantic Search (No Filtering)
# ==============================================================================
print("\n" + "=" * 80)
print("🔍 Step 4: Basic Semantic Search (searching all articles)")
print("=" * 80)

query = "How can I improve my coding skills and learn about algorithms?"
print(f"\nQuery: '{query}'")

# Generate query embedding
query_df = pl.DataFrame({"query": [query]}).with_columns(
    query_embedding=embedding_async(pl.col("query"), provider=Provider.OPENAI)
)

# Search using cosine similarity across ALL articles
basic_results = query_df.join(
    articles_enriched,
    how="cross"
).with_columns(
    similarity=cosine_similarity(pl.col("query_embedding"), pl.col("embedding"))
).sort("similarity", descending=True).head(5)

print("\nTop 5 Results (Basic Search - All Categories):")
for idx, row in enumerate(basic_results.iter_rows(named=True), 1):
    category = row['category'] if row['category'] else 'unknown'
    print(f"{idx}. [{category:12}] {row['title']}")
    print(f"   Similarity: {row['similarity']:.4f}")

# ==============================================================================
# Step 5: Metadata-Enhanced HNSW (Filter by Category First)
# ==============================================================================
print("\n" + "=" * 80)
print("🎯 Step 5: Metadata-Enhanced HNSW Search")
print("=" * 80)

print("\nScenario: User wants technology articles only")
print(f"Query: '{query}'")

# STEP 5A: Filter by metadata (category = technology)
tech_articles = articles_enriched.filter(pl.col("category") == "technology")
print(f"\n→ Filtered to {len(tech_articles)} technology articles (from {len(articles_enriched)} total)")

if len(tech_articles) > 0:
    # STEP 5B: Create HNSW index ONLY for filtered articles
    corpus_embeddings = tech_articles["embedding"].to_list()

    # Create a dataframe with query and corpus for HNSW
    hnsw_search_df = query_df.with_columns(
        corpus_embeddings=pl.lit([corpus_embeddings])
    ).with_columns(
        neighbor_indices=knn_hnsw(
            pl.col("query_embedding"),
            pl.col("corpus_embeddings").list.first(),
            k=min(5, len(tech_articles))  # k cannot exceed corpus size
        )
    )

    # Get the neighbor indices
    neighbor_indices = hnsw_search_df["neighbor_indices"][0]

    print(f"\n→ HNSW found {len(neighbor_indices)} nearest neighbors in technology articles")
    print("\nTop Results (Metadata-Enhanced HNSW - Technology Only):")

    for idx, article_idx in enumerate(neighbor_indices, 1):
        article = tech_articles[article_idx]
        print(f"{idx}. {article['title'][0]}")
        print(f"   Category: {article['category'][0]}, Difficulty: {article['difficulty'][0]}")
else:
    print("\n⚠️  No technology articles found after filtering.")

# ==============================================================================
# Step 6: Advanced Multi-Filter Search
# ==============================================================================
print("\n" + "=" * 80)
print("🔬 Step 6: Advanced Multi-Filter Metadata Search")
print("=" * 80)

query2 = "I want to learn about sustainable energy and environmental solutions"
print(f"\nQuery: '{query2}'")
print("Filters: category='science' AND difficulty='beginner'")

# Generate embedding for new query
query2_df = pl.DataFrame({"query": [query2]}).with_columns(
    query_embedding=embedding_async(pl.col("query"), provider=Provider.OPENAI)
)

# STEP 6A: Apply multiple metadata filters
filtered_articles = articles_enriched.filter(
    (pl.col("category") == "science") &
    (pl.col("difficulty") == "beginner")
)

print(f"\n→ Filtered to {len(filtered_articles)} articles matching criteria")

if len(filtered_articles) > 0:
    # STEP 6B: Search within filtered subset using HNSW
    corpus_embeddings = filtered_articles["embedding"].to_list()

    hnsw_df = query2_df.with_columns(
        corpus_embeddings=pl.lit([corpus_embeddings])
    ).with_columns(
        neighbor_indices=knn_hnsw(
            pl.col("query_embedding"),
            pl.col("corpus_embeddings").list.first(),
            k=min(3, len(filtered_articles))  # k cannot exceed corpus size
        )
    )

    neighbor_indices = hnsw_df["neighbor_indices"][0]

    print(f"\n→ HNSW found {len(neighbor_indices)} relevant articles")
    print("\nRecommended Articles:")

    for idx, article_idx in enumerate(neighbor_indices, 1):
        article = filtered_articles[article_idx]
        print(f"{idx}. {article['title'][0]}")
        print(f"   Category: {article['category'][0]}, Difficulty: {article['difficulty'][0]}")

# ==============================================================================
# Step 7: Performance Comparison
# ==============================================================================
print("\n" + "=" * 80)
print("📈 Step 7: Performance Summary")
print("=" * 80)

print(f"""
Pipeline Execution (with_columns):
  ✓ Taxonomy tagging + Embeddings processed in parallel
  ✓ Total time: {elapsed:.2f}s for {len(articles_enriched)} articles
  ✓ Throughput: {len(articles_enriched)/elapsed:.1f} articles/sec
  ℹ️  Note: Parallel execution achieves ~10% speedup vs sequential
     (Modest gain since embeddings are much faster than taxonomy)

Metadata-Enhanced HNSW Benefits:
  ✓ Filter corpus BEFORE vector search (reduces search space)
  ✓ Ensures results match user constraints (category, difficulty, etc.)
  ✓ Faster search on large corpora (search subset vs. all documents)
  ✓ More relevant results (semantic + metadata matching)

Key Insights:
  • with_columns() executes operations in parallel (best speedup with similar-duration ops)
  • Taxonomy tags provide structured metadata for filtering
  • HNSW provides fast approximate nearest neighbor search
  • Combining both gives best of structured and semantic search
""")

print("=" * 80)
print("✅ Demo completed successfully!")
print("=" * 80)
