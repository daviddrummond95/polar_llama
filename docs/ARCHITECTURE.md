# Polar Llama Architecture

This document provides a comprehensive overview of the Polar Llama architecture, design patterns, and data flow.

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Provider Abstraction](#provider-abstraction)
- [Async Runtime](#async-runtime)
- [Python Integration](#python-integration)
- [Performance Optimizations](#performance-optimizations)

## Overview

Polar Llama is a hybrid Rust/Python library that integrates Large Language Model (LLM) inference capabilities into the Polars dataframe ecosystem. The library is designed for:

- **Performance**: Rust core with async parallel execution
- **Ease of Use**: Python API with Polars expression syntax
- **Flexibility**: Support for multiple LLM providers
- **Type Safety**: Structured outputs with Pydantic validation

## High-Level Architecture

```mermaid
graph TB
    subgraph "Python Layer"
        User[User Code]
        Polars[Polars DataFrame]
        Pydantic[Pydantic Models]
    end

    subgraph "PyO3 Bridge"
        PyExpr[Python Expressions]
        PyModule[polar_llama Module]
    end

    subgraph "Rust Core"
        Expr[Polars Expressions]
        Runtime[Tokio Runtime]
        Client[Model Client Trait]

        subgraph "Providers"
            OpenAI[OpenAI Client]
            Anthropic[Anthropic Client]
            Gemini[Gemini Client]
            Groq[Groq Client]
            Bedrock[Bedrock Client]
        end
    end

    subgraph "External APIs"
        OAPI[OpenAI API]
        AAPI[Anthropic API]
        GAPI[Gemini API]
        GrAPI[Groq API]
        BAPI[AWS Bedrock]
    end

    User -->|DataFrame Operations| Polars
    Polars -->|invoke / pl.col| PyExpr
    PyExpr -->|PyO3 Binding| PyModule
    PyModule -->|Register Expressions| Expr
    Expr -->|Async Execution| Runtime
    Runtime -->|Dispatch| Client
    Client -->|Polymorphism| OpenAI
    Client -->|Polymorphism| Anthropic
    Client -->|Polymorphism| Gemini
    Client -->|Polymorphism| Groq
    Client -->|Polymorphism| Bedrock

    OpenAI -->|HTTPS Request| OAPI
    Anthropic -->|HTTPS Request| AAPI
    Gemini -->|HTTPS Request| GAPI
    Groq -->|HTTPS Request| GrAPI
    Bedrock -->|AWS SDK| BAPI

    Pydantic -.->|Schema Validation| Expr

    OAPI -->|JSON Response| OpenAI
    AAPI -->|JSON Response| Anthropic
    GAPI -->|JSON Response| Gemini
    GrAPI -->|JSON Response| Groq
    BAPI -->|JSON Response| Bedrock

    OpenAI -->|Parse| Runtime
    Anthropic -->|Parse| Runtime
    Gemini -->|Parse| Runtime
    Groq -->|Parse| Runtime
    Bedrock -->|Parse| Runtime

    Runtime -->|Return Series| Expr
    Expr -->|PyO3 Convert| PyModule
    PyModule -->|Return| Polars
    Polars -->|Results| User
```

## Component Architecture

### 1. Python Layer (`Python`)

The user-facing interface where developers interact with Polar Llama through familiar Polars syntax.

```python
# Example: User code
df = pl.DataFrame({'questions': ['What is AI?']})
df = df.with_columns(
    answer=inference_async(
        pl.col('questions'),
        provider=Provider.OPENAI,
        model='gpt-4'
    )
)
```

### 2. PyO3 Bridge (`src/lib.rs`)

The bridge between Python and Rust, using PyO3 for zero-copy interop.

```rust
#[pymodule]
fn polar_llama(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyProvider>()?;
    Ok(())
}
```

**Key Components**:
- Module registration
- Provider enum wrapper
- Expression function exports

### 3. Rust Core

The performance-critical components implemented in Rust.

#### Module Structure

```mermaid
graph LR
    lib[lib.rs] --> expr[expressions.rs]
    lib --> utils[utils.rs]
    lib --> mc[model_client/]

    mc --> trait["mod.rs (ModelClient Trait)"]
    mc --> openai[openai.rs]
    mc --> anthropic[anthropic.rs]
    mc --> gemini[gemini.rs]
    mc --> groq[groq.rs]
    mc --> bedrock[bedrock.rs]

    expr --> msg[chat/message.rs]
```

**File Responsibilities**:

| File | Lines | Responsibility |
|------|-------|----------------|
| `lib.rs` | ~100 | Module exports, PyModule setup, Provider enum |
| `expressions.rs` | ~600 | Polars expression implementations |
| `utils.rs` | ~150 | HTTP utilities, API call helpers |
| `model_client/mod.rs` | ~469 | ModelClient trait, error types, shared logic |
| `model_client/openai.rs` | ~168 | OpenAI API implementation |
| `model_client/anthropic.rs` | ~225 | Anthropic/Claude API implementation |
| `model_client/gemini.rs` | ~182 | Google Gemini API implementation |
| `model_client/groq.rs` | ~166 | Groq API implementation |
| `model_client/bedrock.rs` | ~194 | AWS Bedrock API implementation |
| `chat/message.rs` | ~50 | Message type definitions |

## Data Flow

### Sequential Flow: Single Inference

```mermaid
sequenceDiagram
    participant User
    participant Polars
    participant Expression
    participant Runtime
    participant Client
    participant API

    User->>Polars: df.with_columns(answer=inference(...))
    Polars->>Expression: invoke("inference", column, provider, model)
    Expression->>Runtime: spawn_blocking(async task)
    Runtime->>Client: send_request(messages)
    Client->>Client: format_messages()
    Client->>Client: format_request_body()
    Client->>API: HTTPS POST /v1/chat/completions
    API-->>Client: JSON Response
    Client->>Client: parse_response()
    Client-->>Runtime: String result
    Runtime-->>Expression: String result
    Expression-->>Polars: Series[String]
    Polars-->>User: Updated DataFrame
```

### Parallel Flow: Async Batch Inference

```mermaid
sequenceDiagram
    participant User
    participant Polars
    participant Expression
    participant Runtime
    participant Task1
    participant Task2
    participant TaskN
    participant API

    User->>Polars: df.with_columns(answer=inference_async(...))
    Polars->>Expression: invoke("inference_async", column[N rows])

    Expression->>Runtime: Create tokio::spawn tasks

    par Parallel Execution
        Runtime->>Task1: Process row 1
        Runtime->>Task2: Process row 2
        Runtime->>TaskN: Process row N

        Task1->>API: Request 1
        Task2->>API: Request 2
        TaskN->>API: Request N

        API-->>Task1: Response 1
        API-->>Task2: Response 2
        API-->>TaskN: Response N
    end

    Runtime->>Runtime: futures::join_all(tasks)
    Runtime-->>Expression: Vec[String] results
    Expression-->>Polars: Series[String]
    Polars-->>User: Updated DataFrame
```

## Provider Abstraction

The `ModelClient` trait provides a unified interface for all LLM providers.

```mermaid
classDiagram
    class ModelClient {
        <<trait>>
        +format_messages(messages: Vec~Message~) Value
        +format_request_body(messages: Value, model: String, ...) Value
        +parse_response(response: String) Result~String~
        +send_request(messages: Vec~Message~, ...) Result~String~
    }

    class OpenAIClient {
        +base_url: String
        +api_key: String
        +format_messages()
        +format_request_body()
        +parse_response()
    }

    class AnthropicClient {
        +base_url: String
        +api_key: String
        +format_messages()
        +format_request_body()
        +parse_response()
    }

    class GeminiClient {
        +api_key: String
        +format_messages()
        +format_request_body()
        +parse_response()
    }

    class GroqClient {
        +base_url: String
        +api_key: String
        +format_messages()
        +format_request_body()
        +parse_response()
    }

    class BedrockClient {
        +config: SdkConfig
        +format_messages()
        +format_request_body()
        +parse_response()
    }

    ModelClient <|.. OpenAIClient
    ModelClient <|.. AnthropicClient
    ModelClient <|.. GeminiClient
    ModelClient <|.. GroqClient
    ModelClient <|.. BedrockClient
```

### Trait Methods

1. **`format_messages()`**: Converts internal Message format to provider-specific JSON
2. **`format_request_body()`**: Creates the full request payload
3. **`parse_response()`**: Extracts the text response from provider JSON
4. **`send_request()`**: Orchestrates the full request/response cycle

### Adding a New Provider

To add a new provider:

1. Create `src/model_client/new_provider.rs`
2. Implement the `ModelClient` trait
3. Add to `src/model_client/mod.rs` exports
4. Add enum variant in `src/lib.rs`
5. Update provider dispatch in `src/expressions.rs`

## Async Runtime

Polar Llama uses a global Tokio runtime for efficient async operations.

```rust
use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create Tokio runtime")
});
```

### Why Global Runtime?

- **Performance**: Avoid runtime creation overhead per-request
- **Resource Management**: Single thread pool for all async operations
- **PyO3 Compatibility**: Works around Python GIL constraints

### Async Execution Model

```mermaid
graph TD
    A[Polars Expression] --> B{Sync or Async?}
    B -->|inference| C[RUNTIME.spawn_blocking]
    B -->|inference_async| D[RUNTIME.spawn multiple tasks]

    C --> E[Single tokio::spawn]
    D --> F[Multiple tokio::spawn]

    E --> G[await API call]
    F --> H[futures::join_all]

    H --> I[Collect results]
    G --> J[Return result]
    I --> J
```

## Python Integration

### Expression Registration

Polars expressions are registered at module initialization:

```rust
#[pymodule]
fn polar_llama(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register expressions
    polars_expressions::register_plugin_function::<inference>()?;
    polars_expressions::register_plugin_function::<inference_async>()?;
    // ...
    Ok(())
}
```

### Type Conversions

| Python Type | Rust Type | Conversion |
|-------------|-----------|------------|
| `str` | `String` | Direct |
| `List[Dict]` | `Vec<Message>` | JSON deserialize |
| `polars.Series` | `Series` | PyO3 Polars integration |
| `pydantic.BaseModel` | `serde_json::Value` | JSON schema |
| `Provider` enum | `PyProvider` | PyO3 enum wrapper |

### Pydantic Integration

```mermaid
sequenceDiagram
    participant User
    participant Pydantic
    participant Rust
    participant API

    User->>Pydantic: Define BaseModel schema
    Pydantic->>Rust: Convert to JSON Schema
    Rust->>Rust: Validate with jsonschema crate
    Rust->>API: Request with schema in prompt
    API-->>Rust: JSON response
    Rust->>Rust: Validate against schema
    Rust->>Rust: Convert to Polars Struct
    Rust-->>User: Struct field accessors
```

## Performance Optimizations

### 1. Memory Allocator

**Linux**: Uses jemalloc for better performance

```toml
[target.'cfg(target_os = "linux")'.dependencies]
jemallocator = { version = "0.5", features = ["disable_initial_exec_tls"] }
```

### 2. Link-Time Optimization (LTO)

```toml
[profile.release]
codegen-units = 1
lto = true
```

**Benefits**:
- Smaller binary size
- Better runtime performance
- Cross-crate inlining

### 3. Async Parallel Execution

```rust
// Parallel execution with tokio
let tasks: Vec<_> = rows
    .iter()
    .map(|row| tokio::spawn(async move { process_row(row).await }))
    .collect();

let results = futures::join_all(tasks).await;
```

**Benefits**:
- Non-blocking I/O
- Efficient thread pool usage
- Scales to hundreds of concurrent requests

### 4. Zero-Copy PyO3

PyO3 enables zero-copy data sharing between Python and Rust for Polars Series.

### 5. Lazy Static Runtime

The global Tokio runtime is initialized once and reused across all requests.

## Error Handling

```mermaid
graph TD
    A[API Call] --> B{Success?}
    B -->|Yes| C[parse_response]
    B -->|No| D[HTTP Error]

    C --> E{Valid JSON?}
    E -->|Yes| F[Extract content]
    E -->|No| G[Parse Error]

    F --> H{Schema Valid?}
    H -->|Yes| I[Return Result]
    H -->|No| J[Validation Error]

    D --> K[ModelClientError::HttpError]
    G --> K
    J --> K

    K --> L[Return Error to Polars]
    I --> M[Return Success to Polars]
```

### Error Types

```rust
pub enum ModelClientError {
    HttpError(String),
    ParseError(String),
    ValidationError(String),
    ApiKeyMissing(String),
    // ...
}
```

## Configuration

### Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI | For OpenAI calls |
| `ANTHROPIC_API_KEY` | Anthropic | For Claude calls |
| `GEMINI_API_KEY` | Google | For Gemini calls |
| `GROQ_API_KEY` | Groq | For Groq calls |
| `AWS_ACCESS_KEY_ID` | AWS Bedrock | For Bedrock calls |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock | For Bedrock calls |
| `AWS_REGION` | AWS Bedrock | For Bedrock calls |

### Provider Selection

```python
from polar_llama import Provider

# Available providers
Provider.OPENAI
Provider.ANTHROPIC
Provider.GEMINI
Provider.GROQ
Provider.BEDROCK
```

## Testing Architecture

### Test Organization

```
tests/
├── test_parallel_inference.py    # API integration tests
├── test_imports.py               # Module structure tests
├── test_taxonomy_tagging.py      # Taxonomy feature tests
├── test_structured_outputs.py    # Pydantic schema tests
├── test_message_arrays.py        # Multi-message tests
└── model_client_tests.rs         # Rust unit tests
```

### CI Pipeline

```mermaid
graph LR
    A[Git Push] --> B[Security Audit]
    B --> C[Linux Tests]
    B --> D[Build Wheels]

    C --> E[Python 3.8]
    C --> F[Python 3.9]
    C --> G[Python 3.10]
    C --> H[Python 3.11]

    D --> I[Linux x86_64]
    D --> J[macOS x86_64]
    D --> K[macOS aarch64]
    D --> L[Windows x64]

    I --> M[Release]
    J --> M
    K --> M
    L --> M

    M --> N{Tagged?}
    N -->|Yes| O[Publish to PyPI]
    N -->|No| P[Store Artifacts]
```

## Future Enhancements

1. **Streaming Responses**: Support for streaming LLM outputs
2. **Caching Layer**: Response caching for repeated queries
3. **Batch API Support**: Use provider batch APIs where available
4. **Custom Providers**: Plugin system for custom LLM providers
5. **Observability**: OpenTelemetry integration for tracing

---

**Last Updated**: 2025-12-17
**Version**: 0.2.2
