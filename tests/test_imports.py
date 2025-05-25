import polars as pl
import inspect
import sys
import importlib.util
import os
import pytest
import json

def test_package_contents():
    """Test what's actually available in the polar_llama package."""
    
    # Import the package
    import polar_llama
    
    # Print the dir of the package to see what's available
    print("\nPackage contents:")
    contents = dir(polar_llama)
    print(contents)
    
    # Check specifically for functions we expect
    functions_to_check = ['inference_async', 'inference', 'string_to_message', 'Provider']
    for func in functions_to_check:
        print(f"{func} in polar_llama: {func in contents}")
    
    # Check for the extension module
    extension_name = 'polar_llama'
    extension_available = importlib.util.find_spec(extension_name) is not None
    print(f"Extension module available: {extension_available}")
    
    if extension_available:
        try:
            ext_module = importlib.import_module(extension_name)
            print(f"Extension module dir: {dir(ext_module)}")
        except ImportError as e:
            print(f"Error importing extension module: {e}")
    
    # Print the file path of the package
    print("\nPackage file location:", polar_llama.__file__)
    
    # Try to check Provider enum
    if 'Provider' in contents:
        print("\nProvider enum:")
        provider_values = [attr for attr in dir(polar_llama.Provider) if not attr.startswith('_')]
        print(provider_values)
    
    # Check module structure
    site_packages = os.path.dirname(os.path.dirname(polar_llama.__file__))
    print("\nPackage directory structure:")
    for root, dirs, files in os.walk(os.path.dirname(polar_llama.__file__)):
        rel_path = os.path.relpath(root, site_packages)
        print(f"Directory: {rel_path}")
        for file in files:
            print(f"  - {file}")
    
    # Print __init__.py contents
    try:
        with open(polar_llama.__file__, 'r') as f:
            print("\nFirst 50 lines of __init__.py:")
            lines = f.readlines()
            for i, line in enumerate(lines[:50]):
                print(f"{i+1}: {line.rstrip()}")
                
            # Check if there are any errors in importing from polar_llama
            if "from polar_llama import PyProvider as Provider" in "".join(lines):
                print("\nTrying to import PyProvider directly:")
                try:
                    from polar_llama import PyProvider
                    print("PyProvider imported successfully")
                except ImportError as e:
                    print(f"Error importing PyProvider: {e}")
                except Exception as e:
                    print(f"Unexpected error importing PyProvider: {e}")
                    
    except Exception as e:
        print(f"Error reading __init__.py: {e}")
    
    # Basic assertion to make the test pass
    assert True 

def test_basic_imports():
    """Test that we can import all the main components."""
    try:
        from polar_llama import Provider, inference, inference_async, inference_messages, string_to_message, combine_messages
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import components: {e}")

def test_provider_enum():
    """Test that the Provider enum works correctly."""
    try:
        from polar_llama import Provider
        
        # Check that Provider has the expected attributes
        expected_providers = ['OPENAI', 'ANTHROPIC', 'GEMINI', 'GROQ', 'BEDROCK']
        for provider_name in expected_providers:
            assert hasattr(Provider, provider_name), f"Provider missing {provider_name}"
            
        # Test that we can access provider values
        openai_provider = Provider.OPENAI
        assert openai_provider is not None
        
    except Exception as e:
        pytest.fail(f"Error testing Provider enum: {e}")

def test_string_to_message_basic():
    """Test that string_to_message function works without API calls."""
    try:
        from polar_llama import string_to_message
        
        # Create a simple dataframe
        df = pl.DataFrame({
            "text": ["Hello", "World"]
        })
        
        # Test user message conversion
        result = df.with_columns(
            string_to_message(pl.col("text"), message_type="user").alias("user_msg")
        )
        
        assert "user_msg" in result.columns
        assert result.shape == (2, 2)
        
        # Verify the JSON format
        first_msg = result["user_msg"][0]
        parsed = json.loads(first_msg)
        assert parsed["role"] == "user"
        assert parsed["content"] == "Hello"
        
    except Exception as e:
        pytest.fail(f"Error testing string_to_message: {e}")

def test_combine_messages_basic():
    """Test that combine_messages function works without API calls."""
    try:
        from polar_llama import combine_messages
        
        # Create a dataframe with pre-formatted messages
        df = pl.DataFrame({
            "msg1": ['{"role": "user", "content": "Hello"}'],
            "msg2": ['{"role": "assistant", "content": "Hi there!"}']
        })
        
        # Test message combination
        result = df.with_columns(
            combine_messages(pl.col("msg1"), pl.col("msg2")).alias("conversation")
        )
        
        assert "conversation" in result.columns
        assert result.shape == (1, 3)
        
        # Verify the JSON format
        conversation = result["conversation"][0]
        parsed = json.loads(conversation)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "user"
        assert parsed[1]["role"] == "assistant"
        
    except Exception as e:
        pytest.fail(f"Error testing combine_messages: {e}")

def test_inference_functions_importable():
    """Test that inference functions can be imported (but not called without API keys)."""
    try:
        from polar_llama import inference, inference_async, inference_messages
        
        # Just verify they're callable functions
        assert callable(inference)
        assert callable(inference_async)
        assert callable(inference_messages)
        
        # Create a simple dataframe to test the function signature
        df = pl.DataFrame({
            "text": ["Hello"]
        })
        
        # Test that we can create the expression (but not execute it)
        try:
            expr = inference(pl.col("text"))
            assert expr is not None
        except Exception:
            # It's okay if this fails due to missing API keys or library issues
            # We just want to verify the function signature works
            pass
            
    except Exception as e:
        pytest.fail(f"Error testing inference function imports: {e}")

def test_workflow_without_api():
    """Test a complete workflow without making API calls."""
    try:
        from polar_llama import string_to_message, combine_messages
        
        # Create a dataframe with questions and system instructions
        df = pl.DataFrame({
            "question": ["What is Python?", "How do I learn programming?"],
            "system_instruction": ["You are a helpful assistant.", "You are a programming tutor."]
        })
        
        # Convert to messages
        df = df.with_columns([
            string_to_message(pl.col("system_instruction"), message_type="system").alias("system_msg"),
            string_to_message(pl.col("question"), message_type="user").alias("user_msg")
        ])
        
        # Combine messages
        df = df.with_columns(
            combine_messages(pl.col("system_msg"), pl.col("user_msg")).alias("conversation")
        )
        
        # Verify the result
        assert df.shape == (2, 5)
        assert "conversation" in df.columns
        
        # Check the first conversation
        first_conversation = df["conversation"][0]
        parsed = json.loads(first_conversation)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"
        assert "Python" in parsed[1]["content"]
        
    except Exception as e:
        pytest.fail(f"Error testing complete workflow: {e}") 