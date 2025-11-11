# Polar Llama Test Suite

Comprehensive test suite for polar_llama LLM inference interface.

## Overview

This test suite includes:

- **Rust Integration Tests** (`model_client_tests.rs`): Tests the core Rust model_client interface
- **Python Tests**: Tests the Python/Polars interface exposed via PyO3
  - `test_parallel_inference.py`: Comprehensive tests for parallel execution and provider support
  - `test_structured_outputs.py`: Tests for Pydantic-based structured outputs with schema validation
  - `test_message_arrays.py`: Tests for message array handling
  - `test_provider.py`: Tests for Provider enum
  - Other basic functionality tests

## Features

✅ Tests any subset of providers based on what is configured
✅ Loads API keys from `.env` file
✅ Verifies parallel execution of API calls
✅ Tests both synchronous and asynchronous inference
✅ Tests conversational inference with message arrays
✅ Tests system message support
✅ Tests structured outputs with Pydantic schema validation
✅ Handles edge cases and error conditions

## Setup

### 1. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp ../.env.example ../.env
```

Edit `.env` and add your API keys for the providers you want to test:

```bash
# Add at least one provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
GROQ_API_KEY=gsk_...

# For AWS Bedrock (optional)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

**Note:** Tests will automatically skip providers that don't have API keys configured.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the Rust Library

```bash
cd ..
cargo build --release
```

## Running Tests

### Python Tests

Run all Python tests:

```bash
pytest -v
```

Run specific test file:

```bash
pytest test_parallel_inference.py -v
```

Run with verbose output:

```bash
pytest test_parallel_inference.py -v -s
```

Run tests for a specific provider (if configured):

```bash
pytest test_parallel_inference.py -v -k "openai"
```

Run structured output tests:

```bash
pytest test_structured_outputs.py -v
```

### Rust Tests

Run Rust integration tests:

```bash
cd ..
cargo test --test model_client_tests
```

Run with output:

```bash
cargo test --test model_client_tests -- --nocapture
```

Run specific test:

```bash
cargo test --test model_client_tests test_parallel_execution -- --nocapture
```

## Test Categories

### Rust Tests (`model_client_tests.rs`)

1. **Single Message Inference** - Tests basic inference with each provider
2. **Parallel Execution** - Verifies multiple requests are processed in parallel
3. **Conversation with Message Arrays** - Tests multi-turn conversations
4. **System Message Support** - Tests system message handling
5. **Parallel Execution Timing** - Measures parallel execution performance
6. **Error Handling** - Tests handling of invalid API keys
7. **Provider Enum** - Tests Provider enum functionality

### Python Tests

#### `test_parallel_inference.py`

- **Provider Configuration Tests**
  - Verify Provider enum is available
  - Check at least one provider is configured

- **Single Message Inference Tests**
  - Test simple questions with each configured provider

- **Parallel Execution Tests**
  - Verify multiple questions are processed in parallel
  - Confirm parallel execution is faster than sequential

- **Conversational Inference Tests**
  - Test conversation context is maintained
  - Test system message support

- **Edge Cases**
  - Empty dataframes
  - Null values

#### `test_structured_outputs.py`

- **Basic Structured Output Tests**
  - Test Pydantic model schema conversion
  - Verify LLM returns data matching schema
  - Test Polars Struct field access

- **Multiple Rows Tests**
  - Test structured outputs across multiple DataFrame rows
  - Verify parallel processing with structured schemas
  - Test error handling for individual row failures

- **Schema Validation**
  - Test type coercion (strings, integers, floats, booleans)
  - Test array/list handling
  - Test nested object structures

- **Error Handling**
  - Test `_error`, `_details`, and `_raw` fields
  - Verify graceful handling of validation failures
  - Test that failures in one row don't affect others

#### Other Python Tests

- `test_message_arrays.py` - Tests message array handling
- `test_provider.py` - Tests Provider enum
- `test_imports.py` - Tests module imports
- `test_basic.py` - Basic functionality tests

## Expected Behavior

### Provider Selection

Tests will automatically detect which providers are configured and only run tests for those providers. If no providers are configured, tests will be skipped with a message directing you to add API keys.

### Parallel Execution

The test suite verifies that:
1. Multiple API requests are made concurrently (not sequentially)
2. Parallel execution completes in approximately the time of the slowest request
3. All requests complete successfully (or at least 80% for reliability)

### Output Format

Tests produce detailed output showing:
- Which providers are being tested
- Request timing information
- Sample responses
- Success/failure counts

Example output:
```
🧪 Testing parallel execution with openai
✓ openai processed 5 questions in 2.34s
  Average: 0.47s per question
  Successful: 5/5
```

## Troubleshooting

### No providers configured

If you see: `No providers configured. Add API keys to .env file`

→ Create a `.env` file in the project root with at least one provider's API key

### Import errors

If you see: `Failed to import polar_llama components`

→ Make sure you've built the library: `cargo build --release`
→ Install the library: `pip install -e .` (from project root)

### API rate limits

If tests fail with rate limit errors:

→ Some providers have rate limits on free tiers
→ Try testing one provider at a time
→ Add delays between test runs

### Bedrock tests failing

Bedrock requires:
- AWS credentials configured (CLI, environment, or IAM role)
- Access to Amazon Bedrock service
- Specific models enabled in your AWS account

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use parametrized tests for provider-specific tests
3. Add appropriate skip conditions for providers without keys
4. Include timing information for parallel execution tests
5. Document expected behavior

## License

Same as polar_llama project.
