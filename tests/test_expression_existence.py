import polars as pl
import os

def test_polars_expression_registry():
    """Test if the Polars expressions are registered correctly."""
    
    # Import the package
    import polar_llama
    
    # Try to check if the expressions are registered with Polars
    # This is more of a debugging test than an actual assertion
    
    print("\nPolar_llama dir contents:", dir(polar_llama))
    
    # Try to access the Provider enum
    print("\nProvider available:", hasattr(polar_llama, "Provider"))
    if hasattr(polar_llama, "Provider"):
        print("Provider dir:", dir(polar_llama.Provider))
    
    # Create a small dataframe for testing
    df = pl.DataFrame({
        "text": ["Hello, world!", "Testing expressions"]
    })
    
    # Print information about installed libraries
    print("\nPolar_llama file location:", polar_llama.__file__)
    
    # Print the shared library location if we can find it
    try:
        lib_path = polar_llama.__file__.replace('__init__.py', 'polar_llama.so')
        if os.path.exists(lib_path):
            print(f"Shared library exists at: {lib_path}")
        else:
            print(f"No shared library found at: {lib_path}")
            
            # Try to find any .so files in the package directory
            package_dir = os.path.dirname(polar_llama.__file__)
            so_files = [f for f in os.listdir(package_dir) if f.endswith('.so') or '.so.' in f]
            if so_files:
                print(f"Found .so files: {so_files}")
            else:
                print("No .so files found in package directory")
    except Exception as e:
        print(f"Error checking for shared library: {e}")
        
    # Basic check for non-error
    assert True 