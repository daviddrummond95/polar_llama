### Polar Llama

![Logo](https://raw.githubusercontent.com/daviddrummond95/polar_llama/refs/heads/main/PolarLlama.webp)

#### Overview

Polar Llama is a Python library designed to enhance the efficiency of making parallel inference calls to the ChatGPT API using the Polars dataframe tool. This library enables users to manage multiple API requests simultaneously, significantly speeding up the process compared to serial request handling.

#### Key Features

- **Parallel Inference**: Send multiple inference requests in parallel to the ChatGPT API without waiting for each individual request to complete.
- **Integration with Polars**: Utilizes the Polars dataframe for organizing and handling requests, leveraging its efficient data processing capabilities.
- **Easy to Use**: Simplifies the process of sending queries and retrieving responses from the ChatGPT API through a clean and straightforward interface.
- **Multi-Message Support**: Create and process conversations with multiple messages in context, supporting complex multi-turn interactions.
- **Multiple Provider Support**: Works with OpenAI, Anthropic, Gemini, Groq, and AWS Bedrock models, giving you flexibility in your AI infrastructure.
- **Structured Outputs**: Define response schemas using Pydantic models for type-safe, validated LLM outputs returned as Polars Structs with direct field access.
- **Vector Similarity**: Rust-powered similarity metrics (cosine, dot product, Euclidean distance) for high-performance vector operations.
- **Approximate Nearest Neighbor Search**: HNSW algorithm for fast semantic search and recommendations at scale.

#### Installation

To install Polar Llama, you can use pip:

```bash
pip install polar-llama
```

Alternatively, for development purposes, you can install from source:

```bash
maturin develop
```

#### Example Usage

Here's how you can use Polar Llama to send multiple inference requests in parallel:

```python
import polars as pl
from polar_llama import Provider
import dotenv

dotenv.load_dotenv()

# Example questions
questions = [
    'What is the capital of France?',
    'What is the difference between polars and pandas?'
]

# Creating a dataframe with questions
df = pl.DataFrame({'Questions': questions})

# Using the fluent .llama namespace (recommended)
df = df.with_columns(
    answer=pl.col('Questions').llama.inference_async(
        provider=Provider.OPENAI,
        model='gpt-4o-mini'
    )
)

# Alternative: Using functional API
from polar_llama import string_to_message, inference_async

df = df.with_columns(
    prompt=string_to_message(pl.col("Questions"), message_type='user')
)
df = df.with_columns(
    answer=inference_async(pl.col('prompt'), provider=Provider.OPENAI, model='gpt-4o-mini')
)
```

#### Multi-Message Conversations

Polar Llama supports multi-message conversations, allowing you to maintain context across multiple turns:

```python
import polars as pl
from polar_llama import combine_messages, inference_messages
import dotenv

dotenv.load_dotenv()

# Create a dataframe with system prompts and user questions
df = pl.DataFrame({
    "system_prompt": [
        "You are a helpful assistant.",
        "You are a math expert."
    ],
    "user_question": [
        "What's the weather like today?",
        "Solve x^2 + 5x + 6 = 0"
    ]
})

# Using .llama namespace (recommended)
df = df.with_columns([
    pl.col("system_prompt").llama.to_message(role="system").alias("system_message"),
    pl.col("user_question").llama.to_message(role="user").alias("user_message")
])

# Combine into conversations
df = df.with_columns(
    conversation=combine_messages(pl.col("system_message"), pl.col("user_message"))
)

# Send to model and get responses
df = df.with_columns(
    response=inference_messages(pl.col("conversation"), provider="openai", model="gpt-4")
)
```

#### AWS Bedrock Support

Polar Llama now supports AWS Bedrock models. To use Bedrock, ensure you have AWS credentials configured (via AWS CLI, environment variables, or IAM roles):

```python
import polars as pl
from polar_llama import string_to_message, inference_async
import dotenv

dotenv.load_dotenv()

# Example questions
questions = [
    'What is the capital of France?',
    'Explain quantum computing in simple terms.'
]

# Creating a dataframe with questions
df = pl.DataFrame({'Questions': questions})

# Adding prompts to the dataframe
df = df.with_columns(
    prompt=string_to_message(pl.col("Questions"), message_type='user')
)

# Using AWS Bedrock with Claude model
df = df.with_columns(
    answer=inference_async(pl.col('prompt'), provider='bedrock', model='anthropic.claude-3-haiku-20240307-v1:0')
)
```

#### Structured Outputs with Pydantic

Polar Llama supports structured outputs using Pydantic models. Define your response schema as a Pydantic `BaseModel`, and the LLM will return validated, type-safe data as a Polars Struct:

```python
import polars as pl
from polar_llama import inference_async, Provider
from pydantic import BaseModel

# Define your response schema
class MovieRecommendation(BaseModel):
    title: str
    genre: str
    year: int
    reason: str

# Create a dataframe
df = pl.DataFrame({
    'prompt': ['Recommend a great sci-fi movie from the 2010s']
})

# Get structured output
df = df.with_columns(
    recommendation=inference_async(
        pl.col('prompt'),
        provider=Provider.OPENAI,
        model='gpt-4o-mini',
        response_model=MovieRecommendation
    )
)

# Access struct fields directly!
print(df['recommendation'].struct.field('title')[0])  # "Interstellar"
print(df['recommendation'].struct.field('year')[0])   # 2014
```

**Key Features:**
- **Type Safety**: Responses are validated against your Pydantic schema
- **Direct Field Access**: Use `.struct.field('field_name')` to access individual fields
- **Error Handling**: Built-in `_error`, `_details`, and `_raw` fields for graceful error handling
- **Works Everywhere**: Compatible with `inference_async()`, `inference()`, and `inference_messages()`
- **Multi-Provider**: Works with OpenAI, Anthropic, Groq, Gemini, and Bedrock

**Error Handling:**
```python
# Check for errors in responses
error = df['recommendation'].struct.field('_error')[0]
if error:
    print(f"Error: {error}")
    print(f"Details: {df['recommendation'].struct.field('_details')[0]}")
    print(f"Raw response: {df['recommendation'].struct.field('_raw')[0]}")
```

#### Vector Embeddings

Polar Llama provides parallelized, memory-efficient embedding generation for converting text into vector representations. This is useful for semantic search, clustering, and similarity analysis:

```python
import polars as pl
from polar_llama import embedding_async, Provider

# Create a dataframe with text
df = pl.DataFrame({
    'text': [
        'Machine learning is transforming AI',
        'Natural language processing enables text understanding',
        'Deep learning uses neural networks'
    ]
})

# Generate embeddings in parallel
df = df.with_columns(
    embeddings=embedding_async(
        pl.col('text'),
        provider=Provider.OPENAI,
        model='text-embedding-3-small'  # 1536 dimensions
    )
)

# Access embedding dimensions
df = df.with_columns(
    dimensions=pl.col('embeddings').list.len()
)

print(df['dimensions'][0])  # 1536
```

**Using the .llama namespace:**
```python
df = df.with_columns(
    embeddings=pl.col('text').llama.embedding(
        provider=Provider.OPENAI,
        model='text-embedding-3-small'
    )
)
```

**Supported Models:**
- **OpenAI**: `text-embedding-3-small` (1536 dims), `text-embedding-3-large` (3072 dims)
- **Gemini**: `text-embedding-004` (768 dims)
- **AWS Bedrock**: `amazon.titan-embed-text-v1` (1536 dims)

**Key Features:**
- **Parallel Processing**: All embeddings are generated concurrently using async/await for maximum performance
- **Memory Efficient**: Streaming approach minimizes memory footprint
- **Type Safety**: Returns `List[Float64]` for seamless integration with Polars operations
- **Null Handling**: Gracefully handles null values in input

**Practical Example - Semantic Similarity:**
```python
from polar_llama import cosine_similarity

# Generate embeddings
df = df.with_columns(
    embeddings=embedding_async(pl.col('text'), provider=Provider.OPENAI)
)

# Calculate similarity between first document and all others
query_emb = df["embeddings"][0]
df = df.with_columns(
    similarity=cosine_similarity(
        pl.lit([query_emb]),
        pl.col('embeddings')
    )
)

print(df.select(['text', 'similarity']))
```

#### Vector Similarity and Approximate Nearest Neighbor Search

Polar Llama includes high-performance Rust-powered vector similarity operations and approximate nearest neighbor (ANN) search capabilities for semantic search, recommendations, and clustering:

**Similarity Metrics:**
```python
from polar_llama import cosine_similarity, dot_product, euclidean_distance

df = pl.DataFrame({
    "vec1": [[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
    "vec2": [[1.0, 0.0, 0.0], [2.0, 4.0, 6.0]]
})

# Calculate different similarity metrics
df = df.with_columns([
    cosine_similarity(pl.col("vec1"), pl.col("vec2")).alias("cosine_sim"),
    dot_product(pl.col("vec1"), pl.col("vec2")).alias("dot_prod"),
    euclidean_distance(pl.col("vec1"), pl.col("vec2")).alias("distance")
])

# Using .llama namespace (alternative)
df = df.with_columns(
    cosine_sim=pl.col("vec1").llama.cosine_similarity(pl.col("vec2"))
)
```

**HNSW Approximate Nearest Neighbor Search:**

Fast k-nearest neighbor search using the HNSW (Hierarchical Navigable Small World) algorithm:

```python
from polar_llama import knn_hnsw, embedding_async, Provider

# Create corpus of documents
corpus = pl.DataFrame({
    "doc": ["AI research", "cooking tips", "machine learning", "recipes"]
}).with_columns(
    embedding=embedding_async(pl.col("doc"), provider=Provider.OPENAI)
)

# Create query
query = pl.DataFrame({
    "query": ["artificial intelligence"]
}).with_columns(
    query_emb=embedding_async(pl.col("query"), provider=Provider.OPENAI),
    corpus_emb=pl.lit([corpus["embedding"].to_list()])
).with_columns(
    neighbors=knn_hnsw(
        pl.col("query_emb"),
        pl.col("corpus_emb").list.first(),
        k=2  # Find 2 nearest neighbors
    )
)

# Get nearest neighbor documents
indices = query["neighbors"][0]
print(corpus[indices]["doc"])  # ['AI research', 'machine learning']
```

**Metadata-Enhanced Search:**

Combine taxonomy filtering with vector search for precise, context-aware results:

```python
from polar_llama import tag_taxonomy, embedding_async, knn_hnsw, Provider

# Define taxonomy for metadata
taxonomy = {
    "category": {
        "description": "Content category",
        "values": {
            "technology": "Tech and programming content",
            "cooking": "Food and recipe content"
        }
    }
}

# Use with_columns() for parallel execution
corpus = corpus.with_columns([
    tag_taxonomy(pl.col("content"), taxonomy, provider=Provider.ANTHROPIC)
        .alias("tags"),
    embedding_async(pl.col("content"), provider=Provider.OPENAI)
        .alias("embedding")
])

# Note: with_columns() runs operations in parallel
# Speedup depends on operation durations (best when similar duration)
# Each operation also internally parallelizes all API calls across all documents

# Extract category
corpus = corpus.with_columns(
    category=pl.col("tags").struct.field("category").struct.field("value")
)

# STEP 1: Filter by metadata (category = technology)
tech_docs = corpus.filter(pl.col("category") == "technology")

# STEP 2: Semantic search within filtered subset
query = query.with_columns(
    tech_corpus=pl.lit([tech_docs["embedding"].to_list()])
).with_columns(
    tech_neighbors=knn_hnsw(pl.col("query_emb"), pl.col("tech_corpus").list.first(), k=3)
)

# Results are guaranteed to be technology-related AND semantically relevant
```

**Key Features:**
- **Blazing Fast**: Rust-powered similarity calculations with zero-copy operations
- **Multiple Metrics**: Cosine similarity, dot product, and Euclidean distance
- **HNSW Algorithm**: State-of-the-art approximate nearest neighbor search
- **Parallel Processing**: Vectorized operations for maximum performance
- **Metadata Filtering**: Combine structured and semantic search

See `docs/VECTOR_SIMILARITY_AND_ANN.md` for complete documentation and advanced examples.

#### Benefits

- **Speed**: Processes multiple queries in parallel, drastically reducing the time required for bulk query handling.
- **Scalability**: Scales efficiently with the increase in number of queries, ideal for high-demand applications.
- **Ease of Integration**: Integrates seamlessly into existing Python projects that utilize Polars, making it easy to add parallel processing capabilities.
- **Context Preservation**: Maintain conversation context with multi-message support for more natural interactions.
- **Provider Flexibility**: Choose from multiple LLM providers based on your needs and access.
- **Type Safety**: Get validated, structured outputs using Pydantic schemas for reliable data extraction.

#### Testing

Polar Llama includes a comprehensive test suite that validates parallel execution, provider support, and core functionality.

**Setup:**

1. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your provider API keys
   ```

2. Install test dependencies:
   ```bash
   pip install -r tests/requirements.txt
   ```

**Run Python tests:**
```bash
pytest tests/ -v
```

**Run Rust tests:**
```bash
cargo test --test model_client_tests -- --nocapture
```

Tests automatically detect configured providers and only run tests for those with valid API keys. See [tests/README.md](tests/README.md) for detailed testing documentation.

#### Contributing

We welcome contributions to Polar Llama! If you're interested in improving the library or adding new features, please feel free to fork the repository and submit a pull request.

#### License

Polar Llama is released under the MIT license. For more details, see the LICENSE file in the repository.

#### Roadmap

- [x] **Multi-Message Support**: Support for multi-message conversations to maintain context.
- [x] **Multiple Provider Support**: Support for different LLM providers (OpenAI, Anthropic, Gemini, Groq, AWS Bedrock).
- [x] **Structured Data Outputs**: Add support for structured data outputs using Pydantic models with type validation and Polars Struct returns.
- [ ] **Streaming Responses**: Support for streaming responses from LLM providers.
