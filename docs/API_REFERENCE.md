# Polar Llama API Reference

Complete reference documentation for all Polar Llama expressions and functions.

## Table of Contents

- [Overview](#overview)
- [Polars Namespace (.llama)](#polars-namespace-llama)
- [Providers](#providers)
- [Expressions](#expressions)
  - [inference](#inference)
  - [inference_async](#inference_async)
  - [inference_messages](#inference_messages)
  - [string_to_message](#string_to_message)
  - [combine_messages](#combine_messages)
  - [tag_taxonomy](#tag_taxonomy)
  - [cosine_similarity](#cosine_similarity)
  - [dot_product](#dot_product)
  - [euclidean_distance](#euclidean_distance)
  - [knn_hnsw](#knn_hnsw)
  - [embedding_async](#embedding_async)
- [Data Types](#data-types)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

Polar Llama provides Polars expressions for integrating Large Language Model (LLM) inference directly into your dataframe workflows. All expressions support multiple LLM providers and can be used with standard Polars column operations.

**Two API Styles:**
1. **Fluent `.llama` namespace** (recommended) - More Polars-idiomatic
2. **Functional API** - Direct function calls

### Installation

```bash
pip install polar-llama
```

### Quick Start

```python
import polars as pl
from polar_llama import Provider

df = pl.DataFrame({'questions': ['What is AI?', 'Explain machine learning']})

# Recommended: Using .llama namespace
df = df.with_columns(
    answer=pl.col('questions').llama.inference_async(
        provider=Provider.OPENAI,
        model='gpt-4o-mini'
    )
)
```

## Polars Namespace (.llama)

Polar Llama registers a `.llama` namespace on Polars expressions, providing a fluent, chainable API.

### Available Methods

| Method | Description |
|--------|-------------|
| `.llama.to_message(role='user')` | Convert text to message format |
| `.llama.inference(...)` | Synchronous LLM inference |
| `.llama.inference_async(...)` | Async parallel LLM inference |
| `.llama.tag_taxonomy(taxonomy, ...)` | Taxonomy-based classification |
| `.llama.embedding(...)` | Generate embeddings for text |
| `.llama.cosine_similarity(other)` | Calculate cosine similarity between vectors |
| `.llama.dot_product(other)` | Calculate dot product between vectors |
| `.llama.euclidean_distance(other)` | Calculate Euclidean distance between vectors |

### Examples

**Convert to Message:**
```python
df = df.with_columns(
    msg=pl.col('text').llama.to_message(role='user')
)
```

**Async Inference:**
```python
df = df.with_columns(
    answer=pl.col('question').llama.inference_async(
        provider=Provider.OPENAI,
        model='gpt-4o-mini'
    )
)
```

**Taxonomy Tagging:**
```python
taxonomy = {
    "sentiment": {
        "description": "Emotional tone",
        "values": {
            "positive": "Positive emotions",
            "negative": "Negative emotions"
        }
    }
}

df = df.with_columns(
    tags=pl.col('text').llama.tag_taxonomy(
        taxonomy,
        provider=Provider.ANTHROPIC
    )
)
```

**Benefits:**
- ✅ More readable and chainable
- ✅ Follows Polars conventions (like `.str`, `.dt`, `.list`)
- ✅ IDE autocomplete support
- ✅ Less import noise

## Providers

Polar Llama supports five LLM providers through the `Provider` enum.

### Provider Enum

```python
from polar_llama import Provider

Provider.OPENAI      # OpenAI (GPT-4, GPT-3.5, etc.)
Provider.ANTHROPIC   # Anthropic (Claude)
Provider.GEMINI      # Google Gemini
Provider.GROQ        # Groq
Provider.BEDROCK     # AWS Bedrock
```

### Environment Variables

Configure API keys using environment variables:

| Provider | Environment Variable | Example |
|----------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | `sk-...` |
| Anthropic | `ANTHROPIC_API_KEY` | `sk-ant-...` |
| Gemini | `GEMINI_API_KEY` | `AIza...` |
| Groq | `GROQ_API_KEY` | `gsk_...` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Multiple vars |

### Default Models

Each provider has a default model used when no model is specified:

| Provider | Default Model |
|----------|--------------|
| OpenAI | `gpt-4-turbo` |
| Anthropic | `claude-3-opus-20240229` |
| Gemini | `gemini-1.5-pro` |
| Groq | `llama3-70b-8192` |
| AWS Bedrock | `anthropic.claude-3-haiku-20240307-v1:0` |

## Expressions

### inference

Synchronous LLM inference for single prompts. Processes prompts sequentially.

**Signature:**
```python
def inference(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
) -> pl.Expr
```

**Parameters:**
- `expr` (IntoExpr): The column expression containing prompts to send to the LLM
- `provider` (str | Provider, optional): The LLM provider to use. Defaults to `Provider.OPENAI`
- `model` (str, optional): The specific model name. If not specified, uses the provider's default model

**Returns:**
- `pl.Expr`: Expression that returns a string column with LLM responses

**Example:**
```python
import polars as pl
from polar_llama import inference, Provider

df = pl.DataFrame({
    'prompt': ['What is the capital of France?']
})

df = df.with_columns(
    answer=inference(
        pl.col('prompt'),
        provider=Provider.OPENAI,
        model='gpt-3.5-turbo'
    )
)

print(df)
# shape: (1, 2)
# ┌──────────────────────────────────┬────────────────────┐
# │ prompt                           ┆ answer             │
# │ ---                              ┆ ---                │
# │ str                              ┆ str                │
# ╞══════════════════════════════════╪════════════════════╡
# │ What is the capital of France?   ┆ The capital of...  │
# └──────────────────────────────────┴────────────────────┘
```

**Notes:**
- Processes rows sequentially (not parallel)
- Use `inference_async` for better performance on multiple rows
- Supports all providers and models

---

### inference_async

Asynchronous parallel LLM inference for batch processing. Optimized for high throughput.

**Signature:**
```python
def inference_async(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
) -> pl.Expr
```

**Parameters:**
- `expr` (IntoExpr): The column expression containing prompts to send to the LLM
- `provider` (str | Provider, optional): The LLM provider to use. Defaults to `Provider.OPENAI`
- `model` (str, optional): The specific model name. If not specified, uses the provider's default model
- `response_model` (Type[BaseModel], optional): Pydantic model for structured outputs. When provided, LLM responses are validated against this schema

**Returns:**
- `pl.Expr`: Expression that returns either:
  - String column with LLM responses (when `response_model` is None)
  - Struct column with validated fields (when `response_model` is provided)

**Example 1: Basic Async Inference**
```python
import polars as pl
from polar_llama import inference_async, Provider

df = pl.DataFrame({
    'questions': [
        'What is AI?',
        'What is machine learning?',
        'What is deep learning?'
    ]
})

df = df.with_columns(
    answer=inference_async(
        pl.col('questions'),
        provider=Provider.OPENAI,
        model='gpt-4o-mini'
    )
)

# All three questions are processed in parallel
```

**Example 2: Structured Outputs with Pydantic**
```python
from pydantic import BaseModel
import polars as pl
from polar_llama import inference_async, Provider

class MovieRecommendation(BaseModel):
    title: str
    genre: str
    year: int
    reason: str

df = pl.DataFrame({
    'prompt': ['Recommend a sci-fi movie']
})

df = df.with_columns(
    recommendation=inference_async(
        pl.col('prompt'),
        provider=Provider.OPENAI,
        model='gpt-4o-mini',
        response_model=MovieRecommendation
    )
)

# Access structured fields
print(df['recommendation'].struct.field('title'))
print(df['recommendation'].struct.field('year'))
```

**Performance:**
- Uses Tokio async runtime for parallel execution
- Significantly faster than `inference` for multiple rows
- Ideal for batch processing hundreds or thousands of rows

**Notes:**
- Null values in input are preserved as null in output
- Empty series returns empty result
- JSON schema validation when using `response_model`

---

### inference_messages

Multi-message conversation inference with support for system messages and chat history.

**Signature:**
```python
def inference_messages(
    expr: IntoExpr,
    *,
    provider: Optional[Union[str, Provider]] = None,
    model: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
) -> pl.Expr
```

**Parameters:**
- `expr` (IntoExpr): Column containing JSON arrays of message objects
- `provider` (str | Provider, optional): The LLM provider to use. Defaults to `Provider.OPENAI`
- `model` (str, optional): The specific model name
- `response_model` (Type[BaseModel], optional): Pydantic model for structured outputs

**Returns:**
- `pl.Expr`: Expression that returns LLM responses based on conversation context

**Message Format:**
Each message must be a JSON object with:
- `role`: One of "system", "user", or "assistant"
- `content`: The message text

**Example 1: System Message + User Query**
```python
import polars as pl
from polar_llama import string_to_message, combine_messages, inference_messages, Provider

df = pl.DataFrame({
    'system_prompt': ['You are a helpful math tutor. Keep answers concise.'],
    'user_question': ['What is the derivative of x^2?']
})

# Convert strings to message format
df = df.with_columns([
    string_to_message(pl.col('system_prompt'), message_type='system').alias('system_prompt'),
    string_to_message(pl.col('user_question'), message_type='user').alias('user_question')
])

# Combine into conversation
df = df.with_columns(
    conversation=combine_messages(
        pl.col('system_prompt'),
        pl.col('user_question')
    )
)

# Get response
df = df.with_columns(
    answer=inference_messages(
        pl.col('conversation'),
        provider=Provider.OPENAI,
        model='gpt-4'
    )
)
```

**Example 2: Multi-Turn Conversation**
```python
import polars as pl
from polar_llama import string_to_message, combine_messages, inference_messages

df = pl.DataFrame({
    'msg1': ['What is Python?'],
    'msg2': ['Tell me more about its uses'],
})

# Build conversation history
df = df.with_columns([
    string_to_message(pl.col('msg1'), message_type='user').alias('m1'),
])

# Get first response
df = df.with_columns(
    resp1=inference_messages(pl.col('m1'))
)

# Continue conversation
df = df.with_columns([
    string_to_message(pl.col('resp1'), message_type='assistant').alias('a1'),
    string_to_message(pl.col('msg2'), message_type='user').alias('m2'),
])

df = df.with_columns(
    conversation=combine_messages(pl.col('m1'), pl.col('a1'), pl.col('m2'))
)

df = df.with_columns(
    resp2=inference_messages(pl.col('conversation'))
)
```

**Notes:**
- Supports system messages for controlling model behavior
- Maintains conversation context across multiple turns
- Each row can have a different conversation history
- Messages must be valid JSON

---

### string_to_message

Convert string content into message format for use with `inference_messages`.

**Signature:**
```python
def string_to_message(
    expr: IntoExpr,
    *,
    message_type: str,
) -> pl.Expr
```

**Parameters:**
- `expr` (IntoExpr): Column containing text to convert
- `message_type` (str): The role of the message. Must be one of:
  - `"system"` - System instructions for the model
  - `"user"` - User messages/queries
  - `"assistant"` - Assistant responses (for conversation history)

**Returns:**
- `pl.Expr`: Expression that returns JSON-formatted message strings

**Example:**
```python
import polars as pl

df = pl.DataFrame({
    'user_input': ['What is the weather today?'],
    'system_instruction': ['You are a weather assistant.']
})

df = df.with_columns([
    string_to_message(pl.col('user_input'), message_type='user'),
    string_to_message(pl.col('system_instruction'), message_type='system')
])

print(df['user_input'][0])
# {"role": "user", "content": "What is the weather today?"}
```

**Output Format:**
```json
{
  "role": "system|user|assistant",
  "content": "the message text"
}
```

**Notes:**
- Content is properly JSON-escaped
- Used in conjunction with `combine_messages` and `inference_messages`
- Call directly as a function, not via `.invoke()` method

---

### combine_messages

Combine multiple message expressions into a single JSON array for conversation context.

**Signature:**
```python
def combine_messages(*exprs: IntoExpr) -> pl.Expr
```

**Parameters:**
- `*exprs` (IntoExpr): Variable number of column expressions containing messages. Messages can be:
  - Individual message objects (from `string_to_message`)
  - Arrays of message objects
  - Any valid JSON message format

**Returns:**
- `pl.Expr`: Expression that returns JSON array of all messages in order

**Example:**
```python
import polars as pl
from polar_llama import combine_messages

df = pl.DataFrame({
    'sys': ['{"role": "system", "content": "You are helpful."}'],
    'msg1': ['{"role": "user", "content": "Hello"}'],
    'msg2': ['{"role": "assistant", "content": "Hi there!"}'],
    'msg3': ['{"role": "user", "content": "How are you?"}']
})

df = df.with_columns(
    conversation=combine_messages(
        pl.col('sys'),
        pl.col('msg1'),
        pl.col('msg2'),
        pl.col('msg3')
    )
)

print(df['conversation'][0])
# [{"role": "system", "content": "You are helpful."}, ...]
```

**Features:**
- Handles single messages and message arrays
- Automatically merges nested arrays
- Preserves message order
- Skips empty messages
- Works row-wise (each row gets its own combined array)

**Notes:**
- Required for multi-message conversations
- Can combine system, user, and assistant messages
- Output is ready for `inference_messages`

---

### tag_taxonomy

Advanced taxonomy-based document classification with confidence scores and reasoning.

**Signature:**
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
- `expr` (IntoExpr): Column containing documents to classify
- `taxonomy` (Dict[str, Dict[str, Any]]): Taxonomy definition with structure:
  ```python
  {
      "field_name": {
          "description": "What this field represents",
          "values": {
              "value1": "Definition of value1",
              "value2": "Definition of value2",
              ...
          }
      },
      ...
  }
  ```
- `provider` (str | Provider, optional): The LLM provider to use
- `model` (str, optional): The specific model name

**Returns:**
- `pl.Expr`: Expression that returns a Struct with taxonomy tags. Each field contains:
  - `thinking`: Dict[str, str] - Reasoning for each possible value
  - `reflection`: str - Overall reflection on classification
  - `value`: str - The selected value
  - `confidence`: float - Confidence score (0.0 to 1.0)

**Example 1: Customer Support Classification**
```python
import polars as pl
from polar_llama import tag_taxonomy, Provider

taxonomy = {
    "sentiment": {
        "description": "The emotional tone of the message",
        "values": {
            "positive": "Happy, satisfied, or complimentary",
            "negative": "Angry, frustrated, or complaining",
            "neutral": "Factual without clear emotion"
        }
    },
    "urgency": {
        "description": "How urgent the issue is",
        "values": {
            "high": "Requires immediate attention",
            "medium": "Should be addressed soon",
            "low": "Can wait"
        }
    },
    "category": {
        "description": "The type of support request",
        "values": {
            "billing": "Related to payments or invoices",
            "technical": "Technical issues or bugs",
            "general": "General questions or information"
        }
    }
}

df = pl.DataFrame({
    'message': [
        "URGENT: My account was double charged!",
        "How do I reset my password?",
        "Thanks for the great service!"
    ]
})

df = df.with_columns(
    tags=tag_taxonomy(
        pl.col('message'),
        taxonomy,
        provider=Provider.ANTHROPIC,
        model='claude-3-haiku-20240307'
    )
)

# Access specific fields
print(df['tags'].struct.field('sentiment').struct.field('value'))
# ["negative", "neutral", "positive"]

print(df['tags'].struct.field('urgency').struct.field('confidence'))
# [0.95, 0.75, 0.60]
```

**Example 2: Content Moderation**
```python
taxonomy = {
    "safety": {
        "description": "Content safety assessment",
        "values": {
            "safe": "Content is appropriate for all audiences",
            "sensitive": "May contain mature themes but not harmful",
            "unsafe": "Contains harmful, illegal, or prohibited content"
        }
    },
    "topic": {
        "description": "Primary topic of discussion",
        "values": {
            "politics": "Political discussion or news",
            "technology": "Technology, science, or innovation",
            "entertainment": "Movies, games, or entertainment",
            "other": "Does not fit other categories"
        }
    }
}

df = pl.DataFrame({
    'post': [
        "New breakthrough in quantum computing announced",
        "Debate about healthcare policy continues",
    ]
})

df = df.with_columns(
    classification=tag_taxonomy(pl.col('post'), taxonomy)
)

# Access reasoning
print(df['classification'].struct.field('topic').struct.field('thinking'))
```

**Accessing Results:**

```python
# Get the selected value for a field
df['tags'].struct.field('sentiment').struct.field('value')

# Get confidence score
df['tags'].struct.field('sentiment').struct.field('confidence')

# Get reasoning for specific value
df['tags'].struct.field('sentiment').struct.field('thinking').struct.field('positive')

# Get overall reflection
df['tags'].struct.field('sentiment').struct.field('reflection')

# Filter by confidence threshold
df.filter(
    pl.col('tags').struct.field('urgency').struct.field('confidence') > 0.8
)
```

**Best Practices:**

1. **Clear Definitions**: Provide detailed, non-overlapping definitions for each value
2. **Appropriate Granularity**: 3-7 values per field works best
3. **Model Selection**: Claude models often perform best for reasoning tasks
4. **Confidence Filtering**: Filter results by confidence to ensure quality
5. **Batch Processing**: Use with large datasets for efficient classification

**Use Cases:**
- Customer support ticket routing
- Content moderation and safety
- Email classification and prioritization
- Sentiment analysis with reasoning
- Document categorization
- Multi-label classification

**Notes:**
- Provides detailed reasoning for interpretability
- Confidence scores help assess prediction quality
- Supports hierarchical taxonomies
- More detailed documentation: `docs/TAXONOMY_TAGGING.md`

---

### cosine_similarity

Calculate cosine similarity between two embedding vectors. Measures the cosine of the angle between vectors, producing values from -1 (opposite direction) to 1 (same direction).

**Signature:**
```python
def cosine_similarity(
    vec1: IntoExpr,
    vec2: IntoExpr
) -> pl.Expr
```

**Parameters:**
- `vec1` (IntoExpr): First vector expression (List[Float64])
- `vec2` (IntoExpr): Second vector expression (List[Float64])

**Returns:**
- `pl.Expr`: Expression that returns Float64 cosine similarity scores

**Example:**
```python
import polars as pl
from polar_llama import cosine_similarity, embedding_async, Provider

# Generate embeddings
df = pl.DataFrame({
    "text": ["machine learning", "AI research", "cooking recipes"]
}).with_columns(
    embedding=embedding_async(pl.col("text"), provider=Provider.OPENAI)
)

# Calculate similarity between first doc and all others
query_emb = df["embedding"][0]
df = df.with_columns(
    similarity=cosine_similarity(
        pl.lit([query_emb]),
        pl.col("embedding")
    )
)

# Using .llama namespace (alternative)
df = df.with_columns(
    similarity=pl.col("embedding").llama.cosine_similarity(pl.lit([query_emb]))
)
```

**Properties:**
- **Range**: -1.0 (opposite) to 1.0 (identical)
- **Normalized**: Magnitude-independent (only direction matters)
- **Symmetric**: cosine_similarity(A, B) == cosine_similarity(B, A)

**Use Cases:**
- Text similarity with embeddings
- Document clustering and classification
- Recommendation systems
- Semantic search

**Notes:**
- Vectors must have the same length
- Handles null values (returns null for null input)
- Rust-powered for maximum performance

---

### dot_product

Calculate the dot product of two vectors. Computes the sum of element-wise products.

**Signature:**
```python
def dot_product(
    vec1: IntoExpr,
    vec2: IntoExpr
) -> pl.Expr
```

**Parameters:**
- `vec1` (IntoExpr): First vector expression (List[Float64])
- `vec2` (IntoExpr): Second vector expression (List[Float64])

**Returns:**
- `pl.Expr`: Expression that returns Float64 dot product values

**Example:**
```python
import polars as pl
from polar_llama import dot_product

df = pl.DataFrame({
    "vec1": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "vec2": [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
})

df = df.with_columns(
    dot_prod=dot_product(pl.col("vec1"), pl.col("vec2"))
)

# Result: [32.0, 32.0]  (1*4 + 2*5 + 3*6 = 32)

# Using .llama namespace
df = df.with_columns(
    dot_prod=pl.col("vec1").llama.dot_product(pl.col("vec2"))
)
```

**Properties:**
- **Range**: Unbounded (any real number)
- **Not normalized**: Magnitude affects the result
- **Symmetric**: dot_product(A, B) == dot_product(B, A)
- **Formula**: Σ(a_i × b_i) for i=1 to n

**Use Cases:**
- Neural network operations
- Weighted similarity scores
- Magnitude-aware comparisons
- Linear algebra operations

**Notes:**
- Vectors must have the same length
- Handles null values gracefully
- Zero-copy Rust implementation

---

### euclidean_distance

Calculate the Euclidean distance between two vectors. Computes the straight-line distance in n-dimensional space.

**Signature:**
```python
def euclidean_distance(
    vec1: IntoExpr,
    vec2: IntoExpr
) -> pl.Expr
```

**Parameters:**
- `vec1` (IntoExpr): First vector expression (List[Float64])
- `vec2` (IntoExpr): Second vector expression (List[Float64])

**Returns:**
- `pl.Expr`: Expression that returns Float64 distance values

**Example:**
```python
import polars as pl
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
- **Symmetric**: euclidean_distance(A, B) == euclidean_distance(B, A)
- **Triangle inequality**: dist(A, C) ≤ dist(A, B) + dist(B, C)
- **Formula**: √(Σ(a_i - b_i)²) for i=1 to n

**Use Cases:**
- Spatial data analysis
- K-means clustering
- Anomaly detection
- Distance-based metrics

**Notes:**
- Vectors must have the same length
- Returns 0 for identical vectors
- Efficient Rust implementation

---

### knn_hnsw

Find k-nearest neighbors using the HNSW (Hierarchical Navigable Small World) algorithm for fast approximate nearest neighbor search.

**Signature:**
```python
def knn_hnsw(
    query_expr: IntoExpr,
    reference_expr: IntoExpr,
    *,
    k: int = 5
) -> pl.Expr
```

**Parameters:**
- `query_expr` (IntoExpr): Column containing query embeddings (List[Float64])
- `reference_expr` (IntoExpr): Column containing corpus embeddings (List[List[Float64]])
- `k` (int, optional): Number of nearest neighbors to return. Default: 5

**Returns:**
- `pl.Expr`: Expression that returns List[Int64] containing indices of k-nearest neighbors

**Distance Metric:**
- Uses **cosine distance** = 1 - cosine_similarity
- Optimized for embedding similarity search

**Example 1: Single Query**
```python
import polars as pl
from polar_llama import knn_hnsw, embedding_async, Provider

# Create corpus
corpus = pl.DataFrame({
    "doc": ["AI", "cooking", "ML", "recipes"]
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
        k=2
    )
)

# Get neighbor indices
indices = query["neighbors"][0]
print(corpus[indices]["doc"])  # Nearest neighbors
```

**Example 2: Multiple Queries**
```python
# Multiple queries searching the same corpus
queries = pl.DataFrame({
    "query_text": ["AI research", "cooking tips"],
    "query_emb": [...],  # Query embeddings
    "corpus_emb": [
        corpus["embedding"].to_list(),
        corpus["embedding"].to_list()
    ]
})

result = queries.with_columns(
    neighbors=knn_hnsw(pl.col("query_emb"), pl.col("corpus_emb").list.first(), k=3)
)

# Each row gets its own k-nearest neighbors
```

**Example 3: Metadata-Enhanced Search**
```python
# Filter by metadata BEFORE vector search
tech_docs = corpus.filter(pl.col("category") == "technology")

query = query.with_columns(
    tech_corpus=pl.lit([tech_docs["embedding"].to_list()])
).with_columns(
    neighbors=knn_hnsw(pl.col("query_emb"), pl.col("tech_corpus").list.first(), k=5)
)

# Results guaranteed to be technology-related
```

**Performance:**
- **Time Complexity**: O(log N) search time
- **Accuracy**: Typically >95% recall
- **Scalability**: Efficient for millions of vectors
- **Throughput**: ~500 queries/sec on M1 MacBook Pro

**Use Cases:**
- Large-scale semantic search (>1000 documents)
- Real-time recommendation systems
- Image similarity search
- Any high-dimensional nearest neighbor problem

**Notes:**
- k cannot exceed corpus size (will raise error if too large)
- Empty corpus raises ComputeError
- Each row can search a different corpus
- HNSW index is built per query (optimized for batch queries)

---

### embedding_async

Generate embeddings using various LLM providers. See main documentation above for full details.

**Supported Providers for Embeddings:**
- **OpenAI**: text-embedding-3-small (1536 dims), text-embedding-3-large (3072 dims)
- **Gemini**: text-embedding-004 (768 dims)
- **AWS Bedrock**: amazon.titan-embed-text-v1 (1536 dims)

**Using .llama namespace:**
```python
df = df.with_columns(
    embedding=pl.col("text").llama.embedding(
        provider=Provider.OPENAI,
        model="text-embedding-3-small"
    )
)
```

---

## Data Types

### Input Types

| Python Type | Polars Type | Usage |
|-------------|-------------|-------|
| `str` | `pl.Utf8` | Single prompts, messages |
| `List[Dict]` | `pl.Utf8` (JSON) | Message arrays for conversations |
| `BaseModel` | Schema | Pydantic models for structured outputs |
| `Dict` | `pl.Struct` | Taxonomy definitions |

### Output Types

| Function | Without Schema | With Schema |
|----------|---------------|-------------|
| `inference` | `pl.Utf8` (string) | N/A |
| `inference_async` | `pl.Utf8` (string) | `pl.Struct` (structured) |
| `inference_messages` | `pl.Utf8` (string) | `pl.Struct` (structured) |
| `tag_taxonomy` | `pl.Struct` (taxonomy) | N/A |

### Struct Field Access

```python
# For Pydantic structured outputs
df['result'].struct.field('field_name')

# For taxonomy tags
df['tags'].struct.field('category').struct.field('value')
df['tags'].struct.field('category').struct.field('confidence')
df['tags'].struct.field('category').struct.field('thinking')
df['tags'].struct.field('category').struct.field('reflection')
```

## Error Handling

### Common Errors

**Missing API Key:**
```python
# Error: ModelClientError: API key missing for provider OPENAI
# Solution: Set OPENAI_API_KEY environment variable
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
```

**Invalid Provider:**
```python
# Error: Invalid provider string
# Solution: Use Provider enum or valid provider string
from polar_llama import Provider
provider=Provider.OPENAI  # Correct
provider="openai"         # Also works
```

**Schema Validation Error:**
```python
# Error: Response does not match Pydantic schema
# Solution: Ensure LLM output matches your model definition
# Try a more capable model or adjust schema complexity
```

**Null Values:**
- Null input values → Null output values
- Empty series → Empty series
- Network errors → Error messages in output

### Retry Logic

For production use, implement retry logic:

```python
import polars as pl
from polar_llama import inference_async, Provider
import time

def inference_with_retry(df, col, max_retries=3):
    for attempt in range(max_retries):
        try:
            return df.with_columns(
                result=inference_async(
                    pl.col(col),
                    provider=Provider.OPENAI,
                    model='gpt-4o-mini'
                )
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Examples

### Example 1: Parallel Multi-Provider Comparison

```python
import polars as pl
from polar_llama import inference_async, Provider

df = pl.DataFrame({
    'question': ['What is the meaning of life?']
})

# Get answers from multiple providers in parallel
df = df.with_columns([
    inference_async(pl.col('question'), provider=Provider.OPENAI, model='gpt-4o-mini').alias('openai'),
    inference_async(pl.col('question'), provider=Provider.ANTHROPIC, model='claude-3-haiku-20240307').alias('claude'),
    inference_async(pl.col('question'), provider=Provider.GEMINI, model='gemini-1.5-flash').alias('gemini'),
])

print(df)
```

### Example 2: Sentiment Analysis Pipeline

```python
import polars as pl
from pydantic import BaseModel
from polar_llama import inference_async, Provider

class SentimentResult(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_phrases: list[str]

df = pl.DataFrame({
    'review': [
        'This product is amazing! Best purchase ever.',
        'Terrible quality. Broke after one day.',
        'It works as expected. Nothing special.'
    ]
})

df = df.with_columns(
    analysis=inference_async(
        pl.col('review'),
        provider=Provider.OPENAI,
        model='gpt-4o-mini',
        response_model=SentimentResult
    )
)

# Filter positive reviews with high confidence
positive_reviews = df.filter(
    (pl.col('analysis').struct.field('sentiment') == 'positive') &
    (pl.col('analysis').struct.field('confidence') > 0.8)
)
```

### Example 3: Dynamic Conversation Bot

```python
import polars as pl
from polar_llama import string_to_message, combine_messages, inference_messages, Provider

# Define chatbot personality
system_msg = "You are a friendly AI assistant specializing in Python programming. Keep responses under 100 words."

df = pl.DataFrame({
    'user_query': [
        'How do I read a CSV file?',
        'What about error handling?',
        'Thanks for the help!'
    ]
})

# Add system message
df = df.with_columns(
    system=string_to_message(pl.lit(system_msg), message_type='system')
)

# Convert user queries to messages
df = df.with_columns(
    user_msg=string_to_message(pl.col('user_query'), message_type='user')
)

# Create conversations and get responses
df = df.with_columns(
    conversation=combine_messages(pl.col('system'), pl.col('user_msg'))
)

df = df.with_columns(
    bot_response=inference_messages(
        pl.col('conversation'),
        provider=Provider.ANTHROPIC,
        model='claude-3-sonnet-20240229'
    )
)

print(df.select(['user_query', 'bot_response']))
```

### Example 4: AWS Bedrock with Custom Model

```python
import polars as pl
from polar_llama import inference_async, Provider

# Requires AWS credentials in environment:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

df = pl.DataFrame({
    'prompt': ['Explain quantum entanglement simply']
})

df = df.with_columns(
    answer=inference_async(
        pl.col('prompt'),
        provider=Provider.BEDROCK,
        model='anthropic.claude-3-sonnet-20240229-v1:0'
    )
)
```

### Example 5: Batch Document Summarization

```python
import polars as pl
from polar_llama import inference_async, Provider

# Load 1000 documents
df = pl.read_csv('documents.csv')  # Columns: id, text

# Add summarization prompt
df = df.with_columns(
    prompt=pl.format(
        "Summarize the following in 2-3 sentences:\n\n{}",
        pl.col('text')
    )
)

# Summarize all 1000 documents in parallel
df = df.with_columns(
    summary=inference_async(
        pl.col('prompt'),
        provider=Provider.GROQ,
        model='llama-3.1-70b-versatile'
    )
)

# Groq's fast inference + async execution = very fast summarization
print(df.select(['id', 'summary']))
```

---

## Performance Tips

1. **Use `inference_async` for multiple rows**: 10-100x faster than `inference`
2. **Batch API calls**: Process dataframes with hundreds/thousands of rows at once
3. **Choose appropriate models**: Smaller models (gpt-4o-mini, claude-haiku) for simple tasks
4. **Leverage Groq for speed**: Groq offers fastest inference for supported models
5. **Filter before processing**: Reduce API costs by filtering rows first
6. **Use structured outputs**: Easier parsing than regex/string manipulation
7. **Cache results**: Use Polars' caching or save intermediate results

## Rate Limits

Be aware of provider rate limits:

| Provider | Typical Limit (requests/min) |
|----------|------------------------------|
| OpenAI | 500-10,000 (tier dependent) |
| Anthropic | 50-1,000 (tier dependent) |
| Gemini | 60-1,500 (tier dependent) |
| Groq | 30-100 (free tier) |
| AWS Bedrock | Model dependent |

Use `.limit()` or batch processing to stay within limits:

```python
# Process in batches of 100
for i in range(0, len(df), 100):
    batch = df.slice(i, 100)
    batch = batch.with_columns(
        result=inference_async(pl.col('prompt'))
    )
    # Save or process batch
```

---

## Additional Resources

- **Vector Similarity & ANN Guide**: `docs/VECTOR_SIMILARITY_AND_ANN.md`
- **Taxonomy Tagging Guide**: `docs/TAXONOMY_TAGGING.md`
- **Architecture Overview**: `docs/ARCHITECTURE.md`
- **Examples**: `examples/` directory
  - `similarity_search_demo.py` - Basic semantic search
  - `parallel_embeddings_demo.py` - Performance test with 250 documents
  - `advanced_semantic_search_demo.py` - Metadata-enhanced HNSW
  - `taxonomy_tagging_example.py` - Taxonomy classification
- **Testing Guide**: `tests/README.md`
- **GitHub**: https://github.com/daviddrummond95/polar_llama

---

**Last Updated**: 2025-12-17
**Version**: 0.2.2
