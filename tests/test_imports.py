import polars as pl
import inspect
import sys

def test_package_contents():
    """Test what's actually available in the polar_llama package."""
    
    # Import the package
    import polar_llama
    
    # Print the dir of the package to see what's available
    print("\nPackage contents:")
    contents = dir(polar_llama)
    print(contents)
    
    # Print the __all__ attribute if it exists
    if hasattr(polar_llama, '__all__'):
        print("\n__all__ attribute:")
        print(polar_llama.__all__)
    
    # Print the file path of the package
    print("\nPackage file location:", polar_llama.__file__)
    
    # Try to check Provider enum
    if 'Provider' in contents:
        print("\nProvider enum:")
        provider_values = [attr for attr in dir(polar_llama.Provider) if not attr.startswith('_')]
        print(provider_values)
    
    # Print __init__.py contents
    try:
        with open(polar_llama.__file__, 'r') as f:
            print("\nFirst 50 lines of __init__.py:")
            lines = f.readlines()
            for i, line in enumerate(lines[:50]):
                print(f"{i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"Error reading __init__.py: {e}")
    
    # Basic assertion to make the test pass
    assert True 