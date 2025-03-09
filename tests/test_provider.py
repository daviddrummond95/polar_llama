import polars as pl

def test_provider_enum():
    """Test if the Provider enum is accessible."""
    
    try:
        # Try importing just the Provider enum
        from polar_llama import Provider
        
        # Print the Provider enum
        print("\nProvider enum:", Provider)
        print("Provider enum dir:", dir(Provider))
        
        # Check if we can access specific providers
        providers = [attr for attr in dir(Provider) if not attr.startswith('_')]
        print("Provider values:", providers)
        
        if len(providers) > 0:
            # Try accessing a specific provider
            provider_value = getattr(Provider, providers[0])
            print(f"Provider.{providers[0]}:", provider_value)
        
        # Test passed if we got here
        assert True
        
    except ImportError as e:
        print(f"Import error: {e}")
        assert False, f"Failed to import Provider: {e}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        assert False, f"Unexpected error: {e}" 