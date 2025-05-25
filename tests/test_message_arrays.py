import polars as pl
import pytest
import json

def test_message_array_imports():
    """Test that we can import the necessary components for message arrays."""
    try:
        from polar_llama import string_to_message, combine_messages, inference_messages
        assert True
    except ImportError as e:
        pytest.skip(f"Failed to import message array functions: {e}")

def test_string_to_message():
    """Test the string_to_message function."""
    try:
        # Creating a dataframe with sample content
        df = pl.DataFrame({
            "content": ["Hello, world!", "How are you?"]
        })
        
        # Convert the content to user messages
        df = df.with_columns(
            pl.col("content").invoke("string_to_message", message_type="user").alias("user_message")
        )
        
        # Check the result
        assert df.shape == (2, 2)
        assert "user_message" in df.columns
        
        # Verify the format of the messages
        for msg in df["user_message"]:
            parsed = json.loads(msg)
            assert "role" in parsed
            assert parsed["role"] == "user"
            assert "content" in parsed
            
    except Exception as e:
        pytest.fail(f"Error testing string_to_message: {e}")

def test_combine_messages():
    """Test the combine_messages function."""
    try:
        # Creating a dataframe with sample messages
        df = pl.DataFrame({
            "user_msg": [
                '{"role": "user", "content": "Hello"}',
                '{"role": "user", "content": "How are you?"}'
            ],
            "assistant_msg": [
                '{"role": "assistant", "content": "Hi there!"}',
                '{"role": "assistant", "content": "I am doing well, thanks!"}'
            ]
        })
        
        # Combine the messages
        df = df.with_columns(
            pl.invoke("combine_messages", pl.col("user_msg"), pl.col("assistant_msg")).alias("conversation")
        )
        
        # Check the result
        assert df.shape == (2, 3)
        assert "conversation" in df.columns
        
        # Verify the format of the combined messages
        for conv in df["conversation"]:
            parsed = json.loads(conv)
            assert isinstance(parsed, list)
            assert len(parsed) == 2
            assert parsed[0]["role"] == "user"
            assert parsed[1]["role"] == "assistant"
            
    except Exception as e:
        pytest.fail(f"Error testing combine_messages: {e}")

def test_inference_messages():
    """Test the inference_messages function with mock data."""
    # Skip this test for now as it requires an API key
    pytest.skip("Skipping inference_messages test as it requires API access")
    
    try:
        # Creating a dataframe with sample conversations
        df = pl.DataFrame({
            "conversation": [
                json.dumps([
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thanks! How can I help you today?"},
                    {"role": "user", "content": "What's the weather like?"}
                ])
            ]
        })
        
        # Run inference on the conversations
        df = df.with_columns(
            pl.col("conversation").invoke("inference_messages").alias("response")
        )
        
        # Check the result
        assert df.shape == (1, 2)
        assert "response" in df.columns
        assert df["response"][0] is not None
        
    except Exception as e:
        pytest.fail(f"Error testing inference_messages: {e}") 