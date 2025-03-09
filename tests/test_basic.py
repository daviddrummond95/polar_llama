import polars as pl

def test_basic_import():
    """Test basic import of the package."""
    
    try:
        # Try importing the package
        import polar_llama
        
        # Create a simple dataframe
        df = pl.DataFrame({
            "question": ["What is the capital of France?", "What is the capital of India?"]
        })
        
        # Just verify that we can use polars with our dataframe
        result = df.with_columns(
            question_length=pl.col("question").str.lengths()
        )
        
        # Assert that the dataframe has the expected columns
        assert "question" in result.columns
        assert "question_length" in result.columns
        
        # Print the result for debugging
        print("\nDataFrame Result:")
        print(result)
        
    except ImportError as e:
        print(f"Import error: {e}")
        assert False, f"Failed to import polar_llama: {e}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        assert False, f"Unexpected error: {e}" 