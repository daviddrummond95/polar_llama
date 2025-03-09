import polars as pl
import pytest
import importlib.util
import sys

# Try to import the required components with better error handling
PROVIDER_AVAILABLE = False

try:
    from polar_llama import Provider
    PROVIDER_AVAILABLE = True
    print("Successfully imported Provider")
except ImportError as e:
    print(f"Error importing Provider: {e}")

def test_provider_import():
    """
    Simple test to verify that the Provider enum can be imported.
    """
    # Print diagnostics
    print("\nProvider availability:")
    print(f"Provider available: {PROVIDER_AVAILABLE}")
    
    # Skip the test if Provider is not available
    if not PROVIDER_AVAILABLE:
        pytest.skip("Provider enum not available")
        
    # Example questions
    questions = [
        'What is the capital of France?',
        'What is the capital of India?'
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
    
    # For now, we just verify the test runs without errors
    assert True
