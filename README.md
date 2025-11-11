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
from polar_llama import string_to_message, inference_async, Provider
import dotenv

dotenv.load_dotenv()

# Example questions
questions = [
    'What is the capital of France?',
    'What is the difference between polars and pandas?'
]

# Creating a dataframe with questions
df = pl.DataFrame({'Questions': questions})

# Adding prompts to the dataframe
df = df.with_columns(
    prompt=string_to_message("Questions", message_type='user')
)

# Sending parallel inference requests
df = df.with_columns(
    answer=inference_async('prompt', provider = Provider.OPENAI, model = 'gpt-4o-mini')
)
```

#### Multi-Message Conversations

Polar Llama now supports multi-message conversations, allowing you to maintain context across multiple turns:

```python
import polars as pl
from polar_llama import string_to_message, combine_messages, inference_messages
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

# Convert to structured messages
df = df.with_columns([
    pl.col("system_prompt").invoke("string_to_message", message_type="system").alias("system_message"),
    pl.col("user_question").invoke("string_to_message", message_type="user").alias("user_message")
])

# Combine into conversations
df = df.with_columns(
    pl.invoke("combine_messages", pl.col("system_message"), pl.col("user_message")).alias("conversation")
)

# Send to model and get responses
df = df.with_columns(
    pl.col("conversation").invoke("inference_messages", provider="openai", model="gpt-4").alias("response")
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
    prompt=string_to_message("Questions", message_type='user')
)

# Using AWS Bedrock with Claude model
df = df.with_columns(
    answer=inference_async('prompt', provider='bedrock', model='anthropic.claude-3-haiku-20240307-v1:0')
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
