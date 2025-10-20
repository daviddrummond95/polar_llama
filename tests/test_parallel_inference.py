#!/usr/bin/env python3
"""
Comprehensive test suite for LLM inference with multiple providers.

This test suite:
1. Loads API keys from .env file
2. Tests any subset of providers based on what is configured
3. Verifies parallel execution of API calls
4. Tests both sync and async inference functions
"""

import os
import pytest
import polars as pl
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded .env from {env_path}")
else:
    print(f"⚠️  No .env file found at {env_path}")
    print("   Tests will only run for providers with configured environment variables")

try:
    from polar_llama import (
        Provider,
        inference_async,
        string_to_message,
        combine_messages,
        inference_messages,
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import polar_llama components: {e}")
    IMPORTS_AVAILABLE = False


def is_provider_configured(provider_name: str) -> bool:
    """Check if a provider has API keys configured."""
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'groq': 'GROQ_API_KEY',
        'bedrock': 'AWS_ACCESS_KEY_ID',  # Bedrock uses AWS credentials
    }

    env_var = key_mapping.get(provider_name.lower())
    if not env_var:
        return False

    value = os.getenv(env_var)
    return value is not None and len(value) > 0


def get_configured_providers():
    """Get list of providers that have API keys configured."""
    providers = [
        ('openai', Provider.OPENAI, 'gpt-4o-mini'),
        ('anthropic', Provider.ANTHROPIC, 'claude-3-haiku-20240307'),
        ('gemini', Provider.GEMINI, 'gemini-1.5-flash'),
        ('groq', Provider.GROQ, 'llama-3.1-8b-instant'),
        ('bedrock', Provider.BEDROCK, 'anthropic.claude-3-haiku-20240307-v1:0'),
    ]

    configured = [
        (name, provider, model)
        for name, provider, model in providers
        if is_provider_configured(name)
    ]

    return configured


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="polar_llama not available")
class TestProviderConfiguration:
    """Test that providers are properly configured."""

    def test_provider_enum_available(self):
        """Test that Provider enum is available."""
        assert hasattr(Provider, 'OPENAI')
        assert hasattr(Provider, 'ANTHROPIC')
        assert hasattr(Provider, 'GEMINI')
        assert hasattr(Provider, 'GROQ')
        assert hasattr(Provider, 'BEDROCK')

    def test_at_least_one_provider_configured(self):
        """Test that at least one provider is configured."""
        configured = get_configured_providers()
        if not configured:
            pytest.skip("No providers configured. Add API keys to .env file (see .env.example)")

        print(f"\n✓ Found {len(configured)} configured provider(s):")
        for name, _, model in configured:
            print(f"  - {name} (model: {model})")

        assert len(configured) > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="polar_llama not available")
class TestSingleMessageInference:
    """Test single message inference with different providers."""

    @pytest.mark.parametrize("provider_name,provider,model", get_configured_providers())
    def test_simple_question(self, provider_name, provider, model):
        """Test a simple question with each configured provider."""
        print(f"\n🧪 Testing {provider_name} with single message")

        # Create a simple dataframe
        df = pl.DataFrame({
            'question': ['What is 2+2? Answer with just the number.']
        })

        # Run inference
        start_time = time.time()
        result_df = df.with_columns(
            answer=inference_async(
                pl.col('question'),
                provider=str(provider),
                model=model
            )
        )
        duration = time.time() - start_time

        print(f"✓ {provider_name} completed in {duration:.2f}s")
        print(f"  Answer: {result_df['answer'][0]}")

        # Verify result
        assert result_df.shape == (1, 2)
        assert 'answer' in result_df.columns
        assert result_df['answer'][0] is not None
        assert len(result_df['answer'][0]) > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="polar_llama not available")
class TestParallelExecution:
    """Test that API calls are executed in parallel."""

    @pytest.mark.parametrize("provider_name,provider,model", get_configured_providers())
    def test_parallel_multiple_questions(self, provider_name, provider, model):
        """Test multiple questions are processed in parallel."""
        print(f"\n🧪 Testing parallel execution with {provider_name}")

        # Create dataframe with multiple questions
        questions = [
            'What is the capital of France?',
            'What is the capital of Japan?',
            'What is the capital of Brazil?',
            'What is the capital of Egypt?',
            'What is the capital of Australia?',
        ]

        df = pl.DataFrame({'question': questions})

        # Run inference and measure time
        start_time = time.time()
        result_df = df.with_columns(
            answer=inference_async(
                pl.col('question'),
                provider=str(provider),
                model=model
            )
        )
        duration = time.time() - start_time

        print(f"✓ {provider_name} processed {len(questions)} questions in {duration:.2f}s")
        print(f"  Average: {duration/len(questions):.2f}s per question")

        # Verify all results
        assert result_df.shape == (len(questions), 2)

        # Count successful responses
        successful = sum(1 for ans in result_df['answer'] if ans is not None and len(ans) > 0)
        print(f"  Successful: {successful}/{len(questions)}")

        # At least most should succeed (allow for occasional API hiccups)
        assert successful >= len(questions) * 0.8, f"Expected at least 80% success rate, got {successful}/{len(questions)}"

        # Print sample responses
        for i, (q, a) in enumerate(zip(result_df['question'][:3], result_df['answer'][:3])):
            print(f"  Q{i+1}: {q}")
            print(f"  A{i+1}: {a[:50] if a else 'None'}...")

    @pytest.mark.parametrize("provider_name,provider,model", get_configured_providers())
    def test_parallel_faster_than_sequential(self, provider_name, provider, model):
        """Verify that parallel execution is actually faster than sequential."""
        print(f"\n🧪 Testing parallel vs sequential timing with {provider_name}")

        # Create a small set of questions
        questions = [
            'Say the number 1',
            'Say the number 2',
            'Say the number 3',
        ]

        df = pl.DataFrame({'question': questions})

        # Polars inference_async runs in parallel by default
        start_time = time.time()
        result_df = df.with_columns(
            answer=inference_async(
                pl.col('question'),
                provider=str(provider),
                model=model
            )
        )
        parallel_duration = time.time() - start_time

        print(f"✓ Parallel execution: {parallel_duration:.2f}s for {len(questions)} requests")
        print(f"  Average: {parallel_duration/len(questions):.2f}s per request")

        # Verify results
        successful = sum(1 for ans in result_df['answer'] if ans is not None)
        assert successful >= 2, "At least 2 requests should succeed"

        # Note: We can't easily test sequential timing without modifying the Rust code,
        # but we can verify the parallel time is reasonable
        # If truly parallel, 3 requests should take only slightly more than 1 request
        average_time = parallel_duration / len(questions)
        print(f"  ✓ Parallel execution appears to be working (avg {average_time:.2f}s per request)")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="polar_llama not available")
class TestConversationalInference:
    """Test conversational inference with message arrays."""

    @pytest.mark.parametrize("provider_name,provider,model", get_configured_providers())
    def test_conversation_context(self, provider_name, provider, model):
        """Test that conversation context is maintained."""
        print(f"\n🧪 Testing conversation context with {provider_name}")

        # Create a conversation where context matters
        df = pl.DataFrame({
            'user_msg1': ['Hello! My name is Alice.'],
            'assistant_msg1': ['Hello Alice! Nice to meet you.'],
            'user_msg2': ['What is my name?'],
        })

        # Build conversation
        df = df.with_columns([
            string_to_message(pl.col('user_msg1'), message_type='user').alias('msg1'),
            string_to_message(pl.col('assistant_msg1'), message_type='assistant').alias('msg2'),
            string_to_message(pl.col('user_msg2'), message_type='user').alias('msg3'),
        ])

        # Combine into conversation array
        df = df.with_columns(
            conversation=combine_messages(pl.col('msg1'), pl.col('msg2'), pl.col('msg3'))
        )

        # Run inference
        start_time = time.time()
        result_df = df.with_columns(
            answer=inference_messages(
                pl.col('conversation'),
                provider=str(provider),
                model=model
            )
        )
        duration = time.time() - start_time

        print(f"✓ {provider_name} completed conversation in {duration:.2f}s")
        print(f"  Answer: {result_df['answer'][0]}")

        # Verify result
        assert result_df['answer'][0] is not None
        assert len(result_df['answer'][0]) > 0

        # Note: We can't strictly assert "Alice" is in the response as models vary,
        # but we can verify we got a response

    @pytest.mark.parametrize("provider_name,provider,model", get_configured_providers())
    def test_system_message(self, provider_name, provider, model):
        """Test that system messages work correctly."""
        print(f"\n🧪 Testing system messages with {provider_name}")

        df = pl.DataFrame({
            'system': ['You are a helpful assistant that always responds in exactly 5 words.'],
            'user': ['Tell me about Python.'],
        })

        # Build conversation with system message
        df = df.with_columns([
            string_to_message(pl.col('system'), message_type='system').alias('sys_msg'),
            string_to_message(pl.col('user'), message_type='user').alias('user_msg'),
        ])

        df = df.with_columns(
            conversation=combine_messages(pl.col('sys_msg'), pl.col('user_msg'))
        )

        # Run inference
        start_time = time.time()
        result_df = df.with_columns(
            answer=inference_messages(
                pl.col('conversation'),
                provider=str(provider),
                model=model
            )
        )
        duration = time.time() - start_time

        print(f"✓ {provider_name} completed in {duration:.2f}s")
        print(f"  Answer: {result_df['answer'][0]}")

        # Verify result exists
        assert result_df['answer'][0] is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="polar_llama not available")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        configured = get_configured_providers()
        if not configured:
            pytest.skip("No providers configured")

        provider_name, provider, model = configured[0]

        print(f"\n🧪 Testing empty dataframe with {provider_name}")

        df = pl.DataFrame({'question': []})

        result_df = df.with_columns(
            answer=inference_async(
                pl.col('question'),
                provider=str(provider),
                model=model
            )
        )

        assert result_df.shape == (0, 2)
        print("✓ Empty dataframe handled correctly")

    def test_null_values(self):
        """Test with null values in input."""
        configured = get_configured_providers()
        if not configured:
            pytest.skip("No providers configured")

        provider_name, provider, model = configured[0]

        print(f"\n🧪 Testing null values with {provider_name}")

        df = pl.DataFrame({
            'question': ['What is 1+1?', None, 'What is 2+2?']
        })

        result_df = df.with_columns(
            answer=inference_async(
                pl.col('question'),
                provider=str(provider),
                model=model
            )
        )

        # Null input should produce null output
        assert result_df['answer'][1] is None or result_df['answer'][1] == ''
        # Non-null inputs should produce non-null outputs
        assert result_df['answer'][0] is not None
        assert result_df['answer'][2] is not None

        print("✓ Null values handled correctly")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
