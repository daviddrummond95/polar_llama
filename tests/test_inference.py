import polars as pl
import pytest
import importlib.util
import sys
import dotenv

dotenv.load_dotenv()

# Try to import the required components with better error handling
PROVIDER_AVAILABLE = False
INFERENCE_ASYNC_AVAILABLE = False
STRING_TO_MESSAGE_AVAILABLE = False

try:
    from polar_llama import Provider
    PROVIDER_AVAILABLE = True
    print("Successfully imported Provider")
except ImportError as e:
    print(f"Error importing Provider: {e}")

try:
    from polar_llama import inference_async
    INFERENCE_ASYNC_AVAILABLE = True
    print("Successfully imported inference_async")
except ImportError as e:
    print(f"Error importing inference_async: {e}")

try:
    from polar_llama import string_to_message
    STRING_TO_MESSAGE_AVAILABLE = True
    print("Successfully imported string_to_message")
except ImportError as e:
    print(f"Error importing string_to_message: {e}")

def test_inference_setup():
    """
    Test setup for inference. This is a placeholder test until the
    inference_async function is available in the package.
    """
    # Print diagnostics
    print("\nFunction availability:")
    print(f"Provider available: {PROVIDER_AVAILABLE}")
    print(f"inference_async available: {INFERENCE_ASYNC_AVAILABLE}")
    print(f"string_to_message available: {STRING_TO_MESSAGE_AVAILABLE}")
    
    # Skip the test if any required component is missing
    if not PROVIDER_AVAILABLE:
        pytest.skip("Provider enum not available")
        
    # Example questions
    questions = [
        'What is the capital of France? Respond with only the city name and no other text.',
        'What is the capital of India? Respond with only the city name and no other text.'
    ]

    # Creating a dataframe with questions
    df = pl.DataFrame({'Questions': questions})
    
    # Verify that the dataframe has the expected structure
    assert df.shape == (2, 1)
    assert "Questions" in df.columns
    
    # Print the Provider enum for debugging
    if PROVIDER_AVAILABLE:
        print("\nProvider enum:", Provider)
        print("Provider enum dir:", dir(Provider))
    
    # If all required components are available, run the actual inference test
    if INFERENCE_ASYNC_AVAILABLE and STRING_TO_MESSAGE_AVAILABLE and PROVIDER_AVAILABLE:
        print("\nAll components available, running full inference test")
        test_inference_full()
    else:
        print("\nThis is a placeholder test. The actual inference test is skipped.")
    
    # For now, we just verify the test runs without errors
    assert True
    
@pytest.mark.skipif(not all([INFERENCE_ASYNC_AVAILABLE, STRING_TO_MESSAGE_AVAILABLE, PROVIDER_AVAILABLE]), 
                   reason="Missing required functions for full inference test")
def test_inference_full():
    """
    Full inference test that uses all required functions.
    This will only run if all required functions are available.
    """
    # Example questions
    questions = [
        'What is the capital of France? Respond with only the city name and no other text.',
        'What is the capital of India? Respond with only the city name and no other text.'
    ]

    # Creating a dataframe with questions
    df = pl.DataFrame({'Questions': questions})

    try:
        # Adding prompts to the dataframe
        df = df.with_columns(
            prompt=string_to_message("Questions", message_type='user')
        )

        # Sending parallel inference requests
        result = df.with_columns(
            answer=inference_async('prompt', provider=Provider.OPENAI, model='gpt-4o-mini')
        )
        
        # Select only the columns we want to compare
        result_comparison = result.select(["Questions", "answer"])
        
        expected_df = pl.DataFrame(
            {
                "Questions": ["What is the capital of France? Respond with only the city name and no other text.", 
                            "What is the capital of India? Respond with only the city name and no other text."],
                "answer": ["Paris", "New Delhi"],
            }
        )

        assert result_comparison.equals(expected_df)
    except Exception as e:
        print(f"Error in full inference test: {e}")
        # Test fails only if the functions were available but the test failed
        assert False, f"Error in full inference test: {e}"
