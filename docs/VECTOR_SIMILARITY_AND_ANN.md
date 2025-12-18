# Vector Similarity and Approximate Nearest Neighbor Search

## Overview

Polar Llama provides high-performance vector similarity operations and approximate nearest neighbor (ANN) search capabilities built on Rust. These features enable semantic search, recommendation systems, document clustering, and other vector-based applications at scale.

### Key Features

⚡ **Blazing Fast**: Rust-powered similarity calculations with zero-copy operations

🔍 **Multiple Metrics**: Cosine similarity, dot product, and Euclidean distance

🎯 **HNSW Search**: State-of-the-art approximate nearest neighbor algorithm

📊 **Polars Integration**: Seamless integration with Polars DataFrames

🔄 **Parallel Processing**: All operations vectorized for maximum performance

🎨 **Fluent API**: Available via `.llama` namespace and functional API

## Quick Start

### Basic Cosine Similarity

```python
import polars as pl
from polar_llama import cosine_similarity, Provider, embedding_async

# Create embeddings
df = pl.DataFrame({
    "text": ["machine learning", "artificial intelligence", "cooking recipes"]
})

df = df.with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)

# Calculate similarity between first and other documents
query_emb = df["embedding"][0]
df = df.with_columns(
    similarity=cosine_similarity(
        pl.lit([query_emb]),  # Query embedding
        pl.col("embedding")    # All embeddings
    )
)

print(df.select(["text", "similarity"]))
```

### Approximate Nearest Neighbor Search

```python
from polar_llama import knn_hnsw

# Create corpus and query
corpus_df = pl.DataFrame({
    "doc": ["AI research", "cooking tips", "machine learning", "recipes"],
    "embedding": [[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.15, 0.85]]
})

query_df = pl.DataFrame({
    "query": ["artificial intelligence"],
    "query_emb": [[0.88, 0.12]]
})

# Add corpus embeddings and search
query_df = query_df.with_columns(
    corpus=pl.lit([corpus_df["embedding"].to_list()])
).with_columns(
    neighbors=knn_hnsw(pl.col("query_emb"), pl.col("corpus").list.first(), k=2)
)

# Get neighbor indices
indices = query_df["neighbors"][0]
print(f"Nearest neighbors: {corpus_df[indices]['doc'].to_list()}")
```

## Vector Similarity Metrics

### Cosine Similarity

Measures the cosine of the angle between two vectors, producing values from -1 to 1.

**When to use:**
- Text similarity (with embeddings)
- Recommendation systems
- Document clustering
- Any high-dimensional sparse vectors

**Signature:**
```python
def cosine_similarity(vec1: pl.Expr, vec2: pl.Expr) -> pl.Expr
```

**Example:**
```python
import polars as pl
from polar_llama import cosine_similarity

df = pl.DataFrame({
    "vec1": [[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
    "vec2": [[1.0, 0.0, 0.0], [2.0, 4.0, 6.0]]
})

df = df.with_columns(
    similarity=cosine_similarity(pl.col("vec1"), pl.col("vec2"))
)

# Using .llama namespace (alternative)
df = df.with_columns(
    similarity=pl.col("vec1").llama.cosine_similarity(pl.col("vec2"))
)
```

**Properties:**
- **Range**: -1.0 (opposite) to 1.0 (identical)
- **Normalized**: Magnitude-independent (only direction matters)
- **Symmetric**: cos_sim(A, B) == cos_sim(B, A)

---

### Dot Product

Computes the sum of element-wise products of two vectors.

**When to use:**
- Neural network operations
- Weighted similarity scores
- Magnitude-aware comparisons

**Signature:**
```python
def dot_product(vec1: pl.Expr, vec2: pl.Expr) -> pl.Expr
```

**Example:**
```python
from polar_llama import dot_product

df = pl.DataFrame({
    "vec1": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "vec2": [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
})

df = df.with_columns(
    dot_prod=dot_product(pl.col("vec1"), pl.col("vec2"))
)
# Result: [32.0, 32.0]

# Using .llama namespace
df = df.with_columns(
    dot_prod=pl.col("vec1").llama.dot_product(pl.col("vec2"))
)
```

**Properties:**
- **Range**: Unbounded (can be any real number)
- **Not normalized**: Magnitude affects the result
- **Symmetric**: dot(A, B) == dot(B, A)
- **Formula**: Σ(a_i × b_i) for i=1 to n

---

### Euclidean Distance

Computes the straight-line distance between two points in n-dimensional space.

**When to use:**
- Spatial data analysis
- K-means clustering
- Anomaly detection
- Any distance-based metric

**Signature:**
```python
def euclidean_distance(vec1: pl.Expr, vec2: pl.Expr) -> pl.Expr
```

**Example:**
```python
from polar_llama import euclidean_distance

df = pl.DataFrame({
    "point1": [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
    "point2": [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
})

df = df.with_columns(
    distance=euclidean_distance(pl.col("point1"), pl.col("point2"))
)
# Result: [√3 ≈ 1.732, 5.0]

# Using .llama namespace
df = df.with_columns(
    distance=pl.col("point1").llama.euclidean_distance(pl.col("point2"))
)
```

**Properties:**
- **Range**: 0 (identical) to ∞
- **Symmetric**: dist(A, B) == dist(B, A)
- **Triangle inequality**: dist(A, C) ≤ dist(A, B) + dist(B, C)
- **Formula**: √(Σ(a_i - b_i)²) for i=1 to n

---

## Approximate Nearest Neighbor (ANN) Search

### HNSW Algorithm

Hierarchical Navigable Small World (HNSW) is a graph-based algorithm for fast approximate nearest neighbor search.

**Key Benefits:**
- ⚡ **Fast**: Sub-linear search time O(log N)
- 🎯 **Accurate**: High recall rates (>95% typical)
- 📈 **Scalable**: Efficient for millions of vectors
- 💾 **Memory efficient**: Graph-based structure

**When to use:**
- Large-scale semantic search (>1000 documents)
- Real-time recommendation systems
- Image similarity search
- Any high-dimensional nearest neighbor problem

### knn_hnsw Function

Find k-nearest neighbors using HNSW index.

**Signature:**
```python
def knn_hnsw(
    query_expr: pl.Expr,
    reference_expr: pl.Expr,
    *,
    k: int = 5
) -> pl.Expr
```

**Parameters:**
- `query_expr`: Column containing query embeddings (List[Float64])
- `reference_expr`: Column containing corpus embeddings (List[List[Float64]])
- `k`: Number of nearest neighbors to return (default: 5)

**Returns:**
- List[Int64]: Indices of k-nearest neighbors in the corpus

**Distance Metric:**
- Uses **cosine distance** = 1 - cosine_similarity
- Optimized for embedding similarity search

**Example - Single Query:**
```python
import polars as pl
from polar_llama import knn_hnsw

# Corpus of documents (reference embeddings)
corpus = pl.DataFrame({
    "doc_id": [1, 2, 3, 4],
    "text": ["AI", "cooking", "ML", "recipes"],
    "embedding": [
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.85, 0.15, 0.0],
        [0.15, 0.85, 0.0]
    ]
})

# Query
query = pl.DataFrame({
    "query_emb": [[0.88, 0.12, 0.0]],
    "corpus_emb": [corpus["embedding"].to_list()]
})

# Find 2 nearest neighbors
result = query.with_columns(
    neighbors=knn_hnsw(
        pl.col("query_emb"),
        pl.col("corpus_emb").list.first(),
        k=2
    )
)

indices = result["neighbors"][0]
print(f"Nearest docs: {corpus[indices]['text'].to_list()}")
# Output: ['AI', 'ML']
```

**Example - Multiple Queries:**
```python
# Multiple queries searching the same corpus
queries = pl.DataFrame({
    "query_id": [1, 2],
    "query_emb": [
        [0.88, 0.12, 0.0],  # Tech query
        [0.12, 0.88, 0.0]   # Cooking query
    ],
    "corpus_emb": [
        corpus["embedding"].to_list(),
        corpus["embedding"].to_list()
    ]
})

result = queries.with_columns(
    neighbors=knn_hnsw(pl.col("query_emb"), pl.col("corpus_emb").list.first(), k=2)
)

# Each row gets its own k-nearest neighbors
for idx, neighbors in enumerate(result["neighbors"]):
    print(f"Query {idx+1}: {corpus[neighbors]['text'].to_list()}")
```

---

## Advanced Use Cases

### 1. Metadata-Enhanced Search

Combine taxonomy filtering with vector search for precise results.

```python
import polars as pl
from polar_llama import embedding_async, tag_taxonomy, knn_hnsw, Provider

# Create corpus with content and metadata
corpus = pl.DataFrame({
    "doc": ["Python tutorial", "Cooking guide", "ML course", "Pasta recipe"],
    "content": [
        "Learn Python programming basics...",
        "How to cook delicious meals...",
        "Introduction to machine learning...",
        "Italian pasta cooking instructions..."
    ]
})

# Tag with taxonomy for metadata
taxonomy = {
    "category": {
        "description": "Content category",
        "values": {
            "technology": "Tech and programming",
            "cooking": "Food and recipes"
        }
    }
}

corpus = corpus.with_columns([
    tag_taxonomy(pl.col("content"), taxonomy, provider=Provider.ANTHROPIC)
        .alias("tags"),
    embedding_async(pl.col("content"), provider=Provider.OPENAI)
        .alias("embedding")
])

# Extract category
corpus = corpus.with_columns(
    category=pl.col("tags").struct.field("category").struct.field("value")
)

# STEP 1: Filter by metadata (category = technology)
tech_docs = corpus.filter(pl.col("category") == "technology")

# STEP 2: Semantic search within filtered subset
query = pl.DataFrame({
    "query_text": ["I want to learn about algorithms"]
}).with_columns(
    query_emb=embedding_async(pl.col("query_text"), provider=Provider.OPENAI),
    corpus_emb=pl.lit([tech_docs["embedding"].to_list()])
).with_columns(
    neighbors=knn_hnsw(pl.col("query_emb"), pl.col("corpus_emb").list.first(), k=2)
)

indices = query["neighbors"][0]
print(f"Results: {tech_docs[indices]['doc'].to_list()}")
# Only returns technology documents that are semantically relevant
```

**Benefits:**
- ✅ Guarantees results match metadata constraints
- ✅ Reduces search space for faster queries
- ✅ Combines structured and semantic understanding

---

### 2. Combining Taxonomy + Embedding Generation

Combine taxonomy tagging and embeddings in a single workflow for efficient processing.

```python
import polars as pl
from polar_llama import tag_taxonomy, embedding_async, Provider

df = pl.DataFrame({
    "content": [
        "AI is transforming healthcare...",
        "Best pasta recipes from Italy...",
        # ... more documents
    ]
})

# Use with_columns() for parallel execution of plugin operations
result = df.with_columns([
    tag_taxonomy(pl.col("content"), taxonomy, provider=Provider.ANTHROPIC)
        .alias("tags"),
    embedding_async(pl.col("content"), provider=Provider.OPENAI)
        .alias("embedding")
])
```

**Performance Note:**
- Use `with_columns()` for parallel execution of polar-llama operations
- Polars parallelizes multiple operations, and our async operations use `spawn()` to avoid blocking
- Speedup depends on operation durations:
  - **Two similar operations** (e.g., two taxonomies): ~1.6-1.75x speedup
  - **One fast + one slow** (e.g., taxonomy + embeddings): ~1.1x speedup (10% improvement)
- Each operation also internally parallelizes all API calls across documents for maximum throughput

**Parallel Execution Performance:**
```python
# Two taxonomy operations (similar duration):
df.with_columns([taxonomy1(...), taxonomy2(...)])  # 16.1s (parallel) vs 25.5s (seq) = 1.6x

# Taxonomy + embeddings (different durations):
df.with_columns([taxonomy(...), embeddings(...)])  # 11.5s (parallel) vs 12.7s (seq) = 1.1x

# Best speedup when operations take similar time!
```

---

### 3. Cross-Document Similarity Matrix

Calculate similarity between all pairs of documents.

```python
import polars as pl
from polar_llama import embedding_async, cosine_similarity, Provider

# Create corpus
docs = pl.DataFrame({
    "id": [1, 2, 3],
    "text": ["AI research", "machine learning", "cooking recipes"]
}).with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)

# Cross join to compare all pairs
similarity_matrix = docs.select(["id", "text", "embedding"]).join(
    docs.select(["id", "text", "embedding"]),
    how="cross",
    suffix="_other"
).with_columns(
    similarity=cosine_similarity(pl.col("embedding"), pl.col("embedding_other"))
).select([
    pl.col("id").alias("doc1_id"),
    pl.col("id_other").alias("doc2_id"),
    "text",
    "text_other",
    "similarity"
])

print(similarity_matrix)
```

---

### 4. Clustering with K-Means

Use Euclidean distance for document clustering.

```python
import polars as pl
import numpy as np
from polar_llama import embedding_async, euclidean_distance, Provider
from sklearn.cluster import KMeans

# Generate embeddings
docs = pl.DataFrame({
    "text": [
        "Machine learning tutorial",
        "Cooking pasta",
        "Deep learning guide",
        "Italian recipes",
        "Neural networks",
        "Baking bread"
    ]
}).with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)

# Convert to numpy for sklearn
embeddings_array = np.array(docs["embedding"].to_list())

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings_array)

# Add cluster labels
docs = docs.with_columns(cluster=pl.Series(clusters))

# Find cluster centroids
centroids = pl.DataFrame({
    "cluster": [0, 1],
    "centroid": [kmeans.cluster_centers_[0].tolist(),
                 kmeans.cluster_centers_[1].tolist()]
})

# Calculate distance from each document to its centroid
docs = docs.join(centroids, on="cluster").with_columns(
    distance_to_centroid=euclidean_distance(
        pl.col("embedding"),
        pl.col("centroid")
    )
)

print(docs.group_by("cluster").agg(pl.col("text")))
```

---

## Performance Optimization

### Benchmarks

Tested on M1 MacBook Pro with text-embedding-3-small (1536 dimensions):

| Operation | Documents | Time | Throughput |
|-----------|-----------|------|------------|
| Cosine Similarity | 1,000 pairs | 12ms | 83k pairs/sec |
| Dot Product | 1,000 pairs | 8ms | 125k pairs/sec |
| Euclidean Distance | 1,000 pairs | 10ms | 100k pairs/sec |
| HNSW Search (k=5) | 10,000 corpus | 2ms/query | 500 queries/sec |
| Embedding Generation | 250 docs | 6.5s | 38 docs/sec |

### Best Practices

**1. Batch Operations**
```python
# ✅ Good: Vectorized operation
df.with_columns(
    similarity=cosine_similarity(pl.col("vec1"), pl.col("vec2"))
)

# ❌ Bad: Row-by-row iteration
for row in df.iter_rows():
    # Don't manually calculate similarity
    pass
```

**2. Use HNSW for Large Corpora**
```python
# ✅ Good: HNSW for 1000+ documents
if len(corpus) > 1000:
    use_hnsw = True
else:
    # Exact search with cosine_similarity for small corpora
    use_hnsw = False
```

**3. Filter Before Search**
```python
# ✅ Good: Filter first, then search
filtered_corpus = corpus.filter(pl.col("category") == "tech")
# Search only filtered subset

# ❌ Bad: Search everything, filter after
# Wastes computation on irrelevant documents
```

**4. Parallel Execution with with_columns()**
```python
# ✅ Best: Multiple operations in single with_columns() run in parallel
df = df.with_columns([
    tag_taxonomy(...).alias("tags"),
    embedding_async(...).alias("embedding")
])  # Faster than sequential! (Speedup varies by operation durations)

# ⚠️ Slower: Sequential chaining forces operations to run one after another
df = df.with_columns(tag_taxonomy(...).alias("tags"))
df = df.with_columns(embedding_async(...).alias("embedding"))

# Also good: Lazy evaluation with parallel operations
df = df.lazy().with_columns([
    operation1(...),
    operation2(...)
]).collect()

# Pro tip: Best speedup when operations take similar time (e.g., two taxonomies)
```

---

## API Reference

### Similarity Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `cosine_similarity(vec1, vec2)` | Cosine similarity | Float64 (-1 to 1) |
| `dot_product(vec1, vec2)` | Dot product | Float64 (unbounded) |
| `euclidean_distance(vec1, vec2)` | Euclidean distance | Float64 (0 to ∞) |

### ANN Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `knn_hnsw(query, corpus, k)` | K-nearest neighbors via HNSW | List[Int64] indices |

### Namespace Methods

All functions available via `.llama` namespace:

```python
# Similarity
pl.col("vec1").llama.cosine_similarity(pl.col("vec2"))
pl.col("vec1").llama.dot_product(pl.col("vec2"))
pl.col("vec1").llama.euclidean_distance(pl.col("vec2"))

# Note: knn_hnsw is functional API only (requires multiple columns)
```

---

## Error Handling

### Common Errors

**Vector Length Mismatch:**
```python
# Error: Vectors must have the same length
df = pl.DataFrame({
    "vec1": [[1.0, 2.0, 3.0]],
    "vec2": [[1.0, 2.0]]  # Different length!
})

df.with_columns(
    similarity=cosine_similarity(pl.col("vec1"), pl.col("vec2"))
)
# Raises: "Vectors must have the same length for cosine similarity"
```

**Empty Corpus:**
```python
# Error: Empty corpus for HNSW
corpus = []
result = query_df.with_columns(
    neighbors=knn_hnsw(pl.col("query"), pl.lit([corpus]), k=5)
)
# Raises: ComputeError
```

**k > Corpus Size:**
```python
# Safe: Limit k to corpus size
corpus_size = len(filtered_corpus)
k = min(5, corpus_size)  # Won't exceed available documents

result = query_df.with_columns(
    neighbors=knn_hnsw(pl.col("query"), pl.col("corpus"), k=k)
)
```

### Null Handling

All similarity functions handle null values gracefully:

```python
df = pl.DataFrame({
    "vec1": [[1.0, 0.0], None, [1.0, 1.0]],
    "vec2": [[1.0, 0.0], [0.0, 1.0], None]
})

result = df.with_columns(
    similarity=cosine_similarity(pl.col("vec1"), pl.col("vec2"))
)

# Result:
# Row 0: Valid similarity score
# Row 1: null (vec1 is null)
# Row 2: null (vec2 is null)
```

---

## Examples

Complete runnable examples are available in the `examples/` directory:

1. **`similarity_search_demo.py`**: Basic semantic search with 6 documents
2. **`parallel_embeddings_demo.py`**: Performance test with 250 documents
3. **`advanced_semantic_search_demo.py`**: Metadata-enhanced HNSW and parallel execution

Run examples:
```bash
cd examples
python similarity_search_demo.py
python parallel_embeddings_demo.py
python advanced_semantic_search_demo.py
```

---

## Comparison with Other Libraries

### vs. FAISS

| Feature | Polar Llama | FAISS |
|---------|-------------|-------|
| Integration | Native Polars | Requires conversion |
| Setup | Zero config | Complex setup |
| Learning Curve | Minimal (if you know Polars) | Steep |
| Scale | 10K-1M vectors | 1M-1B vectors |
| Use Case | DataFrame workflows | Large-scale production |

**When to use Polar Llama:**
- Working primarily with Polars DataFrames
- Small to medium scale (< 1M vectors)
- Rapid prototyping and experimentation
- Need metadata filtering before vector search

**When to use FAISS:**
- Very large scale (> 1M vectors)
- Production systems with dedicated vector search
- Need advanced index types (IVF, PQ, etc.)

### vs. Pinecone/Weaviate

| Feature | Polar Llama | Vector Databases |
|---------|-------------|------------------|
| Deployment | In-process (library) | Separate service |
| Cost | Free (open source) | Paid/metered |
| Latency | Microseconds | Network latency |
| Integration | Direct Polars | API calls |
| Use Case | Data pipelines | Production search |

**When to use Polar Llama:**
- Batch processing and analytics
- Data transformation pipelines
- Local development and testing
- Cost-sensitive applications

**When to use Vector Databases:**
- Production user-facing search
- Need persistence and version control
- Distributed/multi-region deployment
- Complex access control

---

## Further Reading

- **Architecture**: `docs/ARCHITECTURE.md` - How Polar Llama works under the hood
- **API Reference**: `docs/API_REFERENCE.md` - Complete API documentation
- **Taxonomy Tagging**: `docs/TAXONOMY_TAGGING.md` - Metadata generation guide
- **HNSW Paper**: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)

---

**Last Updated**: 2025-12-17
**Version**: 0.2.2
