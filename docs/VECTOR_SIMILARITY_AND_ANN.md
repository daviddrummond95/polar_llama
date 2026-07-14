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
| `knn_hnsw(query, corpus, k)` | K-nearest neighbors via HNSW (stateless, rebuilds the graph every call) | List[Int64] indices |
| `HnswIndex.build/add/remove/query/query_one/save/load` | Persistent, incrementally updatable HNSW index (issue #82) -- see [below](#persistent-hnsw-index-hnswindex) | `HnswIndex` / `pl.DataFrame` |

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

## Persistent HNSW Index (`HnswIndex`)

`knn_hnsw` above is **stateless**: every call rebuilds the HNSW graph from
scratch out of whatever corpus column you pass it. That's fine for a
one-off query, but wasteful (and, for a large corpus, slow) if you want to
query the *same* corpus repeatedly, or grow it incrementally over time.

`HnswIndex` (issue #82) is a **persistent, incrementally updatable** index:
build it once, then `.add()`/`.remove()` points into it, `.query()` it as
many times as you like, and `.save()`/`HnswIndex.load()` it to/from disk.
It's a DataFrame-native, stateful, serializable object -- the same shape as
`Checkpoint` and `Codebook` elsewhere in Polar Llama -- not a Polars
expression.

### Why not just rebuild the graph every time?

`instant-distance`'s `HnswMap` (what `knn_hnsw` builds under the hood) is
**immutable** once built -- there is no incremental insert/delete in the
underlying library. `HnswIndex` layers add/remove on top of that immutable
graph itself:

- New/updated points (`.add()`) go into a small brute-force **staging
  buffer**, not the graph -- queryable immediately, without rebuilding
  anything.
- Removed points (`.remove()`) are **soft-deleted** (tombstoned), not
  physically removed -- `.query()` over-fetches from the graph to
  compensate and filters tombstoned ids out of the results.
- Periodically (**compaction**, automatic by default, or via `.compact()`)
  the graph is rebuilt from scratch out of every currently-live point, and
  the staging buffer / tombstones are cleared.

This means a `.add()`/`.remove()` call is cheap (no full rebuild), at the
cost of the staging buffer being brute-force-scanned on every query until
the next compaction -- see [Performance](#performance) below.

### Quick Start

```python
import polars as pl
from polar_llama import HnswIndex, embedding_async, Provider

corpus = pl.DataFrame({
    "doc_id": ["doc-1", "doc-2", "doc-3", "doc-4"],
    "text": ["AI research", "cooking tips", "machine learning", "recipes"],
}).with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)

# Build once.
index = HnswIndex.build(corpus, id_col="doc_id", embedding_col="embedding")

# Query as many times as you like -- no rebuild.
queries = pl.DataFrame({"q": ["artificial intelligence"]}).with_columns(
    embedding=embedding_async(pl.col("q"), provider=Provider.OPENAI)
)
results = index.query(queries, embedding_col="embedding", k=2)
# shape: (2, 4) — query_id | neighbor_id | distance | rank

# Grow the index incrementally -- queryable immediately.
new_docs = pl.DataFrame({
    "doc_id": ["doc-5"],
    "text": ["neural networks"],
}).with_columns(embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI))
index.add(new_docs, id_col="doc_id", embedding_col="embedding")

# Soft-delete.
index.remove(["doc-2"])
assert "doc-2" not in index

# Persist and reload -- identical query results, including any pending
# (not-yet-compacted) staging/tombstone state.
index.save("my_index.bin")
reloaded = HnswIndex.load("my_index.bin")
```

### API

```python
HnswIndex.build(df, id_col, embedding_col, *, ef_construction=200, ef_search=200,
                 seed=42, auto_compact=True, compact_staged_min=1000,
                 compact_staged_ratio=0.10, compact_tombstone_min=1000,
                 compact_tombstone_ratio=0.10) -> HnswIndex
HnswIndex.load(path) -> HnswIndex

index.add(df, id_col, embedding_col) -> int        # rows upserted
index.remove(ids: list[str] | pl.Series) -> int    # rows soft-deleted
index.compact() -> None                            # force a rebuild now

index.query(query_df, embedding_col, k=10) -> pl.DataFrame  # query_id | neighbor_id | distance | rank
index.query_one(vector: list[float], k=10) -> pl.DataFrame  # neighbor_id | distance | rank
index.knn(expr, k=10) -> pl.Expr                    # List[Struct{neighbor_id, distance, rank}] per row

index.save(path) -> None
len(index)              # live point count
index.dim               # embedding dimensionality
"some_id" in index       # membership check
index.staged_len         # pending (not-yet-compacted) adds
index.tombstone_len      # pending (not-yet-compacted) removals
index.should_compact     # would `.compact()` do real work right now?
index.auto_compact       # get/set
```

`.query()`'s `query_id` column is `query_df`'s **0-based row index** --
there's no separate id parameter for queries, only for the indexed corpus
(`id_col` on `.build()`/`.add()`).

`.knn()` bridges `.query()`'s batch core into a lazy expression via
`map_batches`, for use inside `with_columns`:

```python
queries.with_columns(
    neighbors=index.knn(pl.col("embedding"), k=5)
).explode("neighbors").unnest("neighbors")
```

### Upsert and soft-delete semantics

- **Upsert**: `.add()` with an id already in the index replaces its vector
  -- the old entry is tombstoned (never mutated in place, since the main
  graph is immutable) and the new one goes into staging. A duplicate id
  *within* one `.build()`/`.add()` call resolves last-row-wins.
- **Soft-delete**: `.remove()` on an id not currently live (unknown, or
  already removed) is a silent no-op. `k` neighbors are still returned
  after a removal as long as the index has that many *live* points left --
  `.query()`'s over-fetch (`k' = min(k + tombstone_count, ef_search)`)
  exists specifically so removed points don't silently shrink your result
  count.

### Compaction

`auto_compact=True` (the default) triggers `.compact()` automatically,
after `.add()`/`.remove()`, once the staging buffer or the tombstone count
exceeds `max(compact_*_min, compact_*_ratio * len(index))`. Set
`auto_compact=False` to batch many `.add()`/`.remove()` calls and
`.compact()` once yourself (cheaper than compacting after every call);
`index.should_compact` tells you whether a threshold is currently crossed.

### Performance

- **Query**: sub-millisecond to a few milliseconds per query against a
  compacted 100k-point index (approximate HNSW search, same complexity
  class as `knn_hnsw`); each un-compacted staged point adds one brute-force
  distance computation per query, so a large pending staging buffer
  (thousands of points) is the main way to slow queries down between
  compactions.
- **`.save()`/`.load()`**: a low-single-digit number of seconds for a
  100k-point index (dominated by `bincode` (de)serializing the graph).
  `instant-distance` has no mmap or lazy-load path, so `.load()` always
  deserializes the whole graph into memory up front -- there is no
  partial/streaming load for indexes too large to fit in RAM.
- See `scripts/bench_hnsw_index.py` for a runnable 100k-point benchmark
  (not part of the test suite -- it's slow by design).

### When to use `HnswIndex` vs. `knn_hnsw`

| | `knn_hnsw` | `HnswIndex` |
|---|---|---|
| Corpus | Rebuilt every call | Built once, updated incrementally |
| State | None (pure expression) | Stateful object (build/add/remove/save/load) |
| Best for | One-off query against a corpus you already have in memory | A corpus you query repeatedly, or that grows over time |
| Persistence | None | `.save()`/`.load()` |

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

**Last Updated**: 2026-07-14
**Version**: 0.8.0
