#!/usr/bin/env python3
"""
Demo: Semantic similarity search using embeddings, cosine similarity, and HNSW.

This example demonstrates:
1. Generating embeddings with OpenAI
2. Calculating cosine similarity between embeddings
3. Using HNSW for fast approximate nearest neighbor search
"""

import os
from pathlib import Path
import polars as pl
from polar_llama import embedding_async, cosine_similarity, knn_hnsw

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("⚠️  dotenv not installed, using existing environment variables")

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set. Please set it to run this demo.")
    print("   Create a .env file in the project root with: OPENAI_API_KEY=your-key")
    exit(1)

print("=" * 70)
print("Semantic Similarity Search Demo")
print("=" * 70)

# Step 1: Create a corpus of documents
print("\n1️⃣  Creating corpus of documents...")
corpus = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "text": [
        "Python is a high-level programming language",
        "Machine learning uses algorithms to find patterns",
        "Neural networks are inspired by the human brain",
        "Data science involves statistics and programming",
        "The weather is sunny today",
        "Cats are popular pets around the world",
    ]
})
print(corpus)

# Step 2: Generate embeddings for the corpus
print("\n2️⃣  Generating embeddings for corpus (this may take a few seconds)...")
from polar_llama import Provider
corpus_with_embeddings = corpus.with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)
print(f"✓ Generated {len(corpus_with_embeddings)} embeddings")
print(f"  Embedding dimensions: {corpus_with_embeddings['embedding'][0].__len__()}")

# Step 3: Create query embeddings
print("\n3️⃣  Creating query...")
query = pl.DataFrame({
    "query": ["deep learning and artificial intelligence"]
})

query_with_embedding = query.with_columns(
    query_embedding=embedding_async(pl.col("query"), provider=Provider.OPENAI)
)
print(f"Query: '{query['query'][0]}'")

# Step 4: Calculate cosine similarity with all documents
print("\n4️⃣  Calculating cosine similarity with all documents...")

# Cross join to compare query with all corpus documents
similarities = query_with_embedding.join(
    corpus_with_embeddings,
    how="cross"
).with_columns(
    similarity=cosine_similarity(
        pl.col("query_embedding"),
        pl.col("embedding")
    )
).sort("similarity", descending=True)

print("\nTop 3 most similar documents:")
print(similarities.select(["id", "text", "similarity"]).head(3))

# Step 5: Demonstrate HNSW approximate nearest neighbor search
print("\n5️⃣  Using HNSW for fast approximate nearest neighbor search...")

# For HNSW, we need the corpus embeddings as a list
# Create a structure where we have the query embedding and all corpus embeddings
hnsw_df = query_with_embedding.with_columns(
    corpus_embeddings=pl.lit([corpus_with_embeddings["embedding"].to_list()])
).with_columns(
    nearest_neighbors=knn_hnsw(
        pl.col("query_embedding"),
        pl.col("corpus_embeddings").list.first(),
        k=3
    )
)

neighbor_indices = hnsw_df["nearest_neighbors"][0]
print(f"\nHNSW found {len(neighbor_indices)} nearest neighbors:")
print(f"Neighbor indices: {neighbor_indices}")

# Show the actual documents
print("\nNearest neighbor documents:")
for idx in neighbor_indices:
    doc = corpus.filter(pl.col("id") == idx + 1)  # +1 because indices are 0-based
    if len(doc) > 0:
        print(f"  [{idx}] {doc['text'][0]}")

# Step 6: Compare multiple queries
print("\n6️⃣  Comparing multiple queries...")
multi_query = pl.DataFrame({
    "query": [
        "programming languages",
        "artificial intelligence",
        "animals and pets"
    ]
})

multi_query_embeddings = multi_query.with_columns(
    query_embedding=embedding_async(pl.col("query"), provider=Provider.OPENAI)
)

# Calculate similarity for each query using cross join
for i in range(len(multi_query)):
    query_row = multi_query_embeddings.filter(pl.int_range(pl.len()) == i)
    query_text = query_row["query"][0]

    # Cross join query with corpus
    sims = query_row.join(corpus_with_embeddings, how="cross").with_columns(
        similarity=cosine_similarity(
            pl.col("query_embedding"),
            pl.col("embedding")
        )
    ).sort("similarity", descending=True)

    top_match = sims.head(1)
    print(f"\nQuery: '{query_text}'")
    print(f"  → Most similar: '{top_match['text'][0]}'")
    print(f"  → Similarity: {top_match['similarity'][0]:.4f}")

print("\n" + "=" * 70)
print("✅ Demo completed successfully!")
print("=" * 70)
